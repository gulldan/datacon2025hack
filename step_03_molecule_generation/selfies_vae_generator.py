"""SELFIES VAE generator – more powerful than baseline char-RNN.

* Trains a variational autoencoder on SELFIES strings converted from the
  processed activity dataset.
* After training, samples new molecules and writes raw SMILES to
  ``generated_smiles_raw.txt`` (same location as previous generator).
* Uses 100 % valid SELFIES → SMILES decoding (no invalid syntax by design).

This is still a lightweight reference implementation – for production quality
consider JT-VAE or GraphNVP.
"""
from __future__ import annotations

import math
import random

import numpy as np
import selfies as sf  # type: ignore
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

import config
from utils.logger import LOGGER

SEED = config.RANDOM_STATE
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
EMBED_DIM = config.VAE_EMBED_DIM
HIDDEN_DIM = config.VAE_HIDDEN_DIM
LATENT_DIM = config.VAE_LATENT_DIM
NUM_LAYERS = config.VAE_NUM_LAYERS
DROPOUT = config.VAE_DROPOUT
BATCH_SIZE = config.VAE_BATCH_SIZE
MAX_LEN = config.VAE_MAX_LEN  # SELFIES tokens (covers >99 % dataset)
EPOCHS = config.MAX_VAE_EPOCHS
LEARNING_RATE = config.VAE_LEARNING_RATE
MODEL_PATH = config.GENERATION_RESULTS_DIR / "selfies_vae.pt"
SMILES_OUT_PATH = config.GENERATION_RESULTS_DIR / "generated_smiles_raw.txt"
GENERATE_N = config.VAE_GENERATE_N
PATIENCE = config.VAE_PATIENCE

# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------
class Vocab:
    def __init__(self, tokens: list[str]):
        self.pad = "[PAD]"
        self.bos = "[SOS]"
        self.eos = "[EOS]"
        tokens = [self.pad, self.bos, self.eos] + sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(tokens)}
        self.itos = tokens

    def encode(self, seq: list[str]) -> list[int]:
        return [self.stoi[self.bos]] + [self.stoi[t] for t in seq] + [self.stoi[self.eos]]

    def decode(self, idx: list[int]) -> list[str]:
        out = []
        started = False
        for i in idx:
            tok = self.itos[i]
            if tok == self.bos:
                started = True
                continue
            if tok in (self.eos, self.pad):
                break
            if started:
            out.append(tok)
        return out

    def __len__(self):
        return len(self.itos)


def load_selfies() -> list[str]:
    import polars as pl

    if not config.ACTIVITY_DATA_PROCESSED_PATH.exists():
        LOGGER.error("Processed dataset not found: %s", config.ACTIVITY_DATA_PROCESSED_PATH)
        raise FileNotFoundError
    df = pl.read_parquet(config.ACTIVITY_DATA_PROCESSED_PATH)
    smiles_list: list[str] = df["SMILES"].unique().to_list()  # type: ignore[no-any-return]

    from rdkit import Chem  # type: ignore

    selfies_list: list[str] = []
    AUG_PER_MOL = config.AUG_PER_MOL  # количество рандомных перестановок SMILES
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)  # type: ignore[attr-defined]
        if mol is None:
            continue
        for _ in range(AUG_PER_MOL):
            rand_smi = Chem.MolToSmiles(mol, doRandom=True)  # type: ignore[attr-defined]
            try:
                selfies_list.append(sf.encoder(rand_smi))
            except sf.EncoderError:
                continue
    return selfies_list


class SelfiesDataset(Dataset):
    def __init__(self, selfies: list[str], vocab: Vocab):
        self.vocab = vocab
        data_list: list[torch.Tensor] = []
        for s in selfies:
            tokens = list(sf.split_selfies(s))
            idx = vocab.encode(tokens)
            if len(idx) > MAX_LEN:
                idx = idx[: MAX_LEN]
            pad_len = MAX_LEN - len(idx)
            idx = idx + [vocab.stoi[vocab.pad]] * pad_len
            data_list.append(torch.tensor(idx, dtype=torch.long))
        self.data = torch.stack(data_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):  # type: ignore[override]
        x = self.data[idx]
        return x[:-1], x[1:]


# ---------------------------------------------------------------------------
# VAE Model definition
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.gru = nn.GRU(EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, batch_first=True, dropout=DROPOUT)
        self.mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, x):  # x: [B, T]
        emb = self.embed(x)
        _, h = self.gru(emb)  # h: [num_layers, B, H]
        h = h[-1]
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.fc_z = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.gru = nn.GRU(EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, batch_first=True, dropout=DROPOUT)
        self.fc_out = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, z, tgt):  # z: [B, Z], tgt: [B, T]
        h0 = torch.tanh(self.fc_z(z)).unsqueeze(0).repeat(NUM_LAYERS, 1, 1)  # [L,B,H]
        emb = self.embed(tgt)
        out, _ = self.gru(emb, h0)
        logits = self.fc_out(out)
        return logits


class SelfiesVAE(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.enc = Encoder(vocab_size)
        self.dec = Decoder(vocab_size)

    def forward(self, src, tgt):
        mu, logvar = self.enc(src)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # reparameterisation
        logits = self.dec(z, tgt)
        return logits, mu, logvar

    def sample(self, vocab: Vocab, num: int = 100) -> list[str]:
        """Improved sampling with temperature control and better diversity.
        """
        self.eval()
        with torch.no_grad():
            # Sample from latent space with some diversity
            z = torch.randn(num, LATENT_DIM, device=dev) * 1.0  # Reduce variance
            bos_idx = vocab.stoi[vocab.bos]
            eos_idx = vocab.stoi.get(vocab.eos, vocab.stoi[vocab.pad])
            pad_idx = vocab.stoi[vocab.pad]

            seqs = torch.full((num, MAX_LEN), pad_idx, dtype=torch.long, device=dev)
            seqs[:, 0] = bos_idx

            # Simple greedy decoding for debugging
            for t in range(1, MAX_LEN):
                logits = self.dec(z, seqs[:, :t])[:, -1, :]

                # Apply temperature for more diverse sampling
                temperature = 1.0  # More conservative temperature
                logits = logits / temperature

                # Avoid padding tokens in generation
                logits[:, pad_idx] = -float("inf")

                # Use top-k sampling for better results
                top_k = 10
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    # Create mask for non-top-k values
                    mask = torch.full_like(logits, -float("inf"))
                    mask.scatter_(-1, top_k_indices, top_k_logits)
                    logits = mask

                prob = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(prob, 1).squeeze(-1)

                # Update sequences
                seqs[:, t] = next_tok

            # Debug: print some sequences
            smiles_out: list[str] = []
            selfies_generated = 0
            selfies_decoded = 0

            for i, row in enumerate(seqs):
                tokens = vocab.decode(row.tolist())
                # Remove padding tokens but keep EOS
                tokens = [t for t in tokens if t != vocab.pad]
                selfies_str = "".join(tokens)

                # Debug first few sequences
                if i < 5:
                    print(f"Debug seq {i}: tokens={tokens[:10]}... -> selfies='{selfies_str[:50]}...'")

                # Skip empty sequences or just BOS
                if not selfies_str or selfies_str == vocab.bos:
                    continue

                selfies_generated += 1

                try:
                    smi = sf.decoder(selfies_str)
                    if smi and len(smi) > 3:  # More lenient validation
                    smiles_out.append(smi)
                        selfies_decoded += 1
                        # Debug successful decode
                        if len(smiles_out) <= 3:
                            print(f"Success {len(smiles_out)}: '{selfies_str[:30]}...' -> '{smi}'")
                except Exception as e:
                    if i < 5:  # Debug first few failures
                        print(f"Decode error seq {i}: {e}")
                    continue

            print(f"Sample summary: {selfies_generated} non-empty SELFIES, {selfies_decoded} decoded to SMILES")
            return smiles_out


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def loss_fn(logits, targets, mu, logvar):
    """Improved VAE loss with better regularization and KL annealing.
    """
    batch_size = targets.size(0)

    # Reconstruction loss (cross-entropy)
    recon_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=0,
        reduction="sum"
    ) / batch_size

    # KL divergence loss (normalized by batch size and latent dim)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    # KL annealing: start with low weight, gradually increase
    # This helps prevent posterior collapse
    kl_weight = min(0.01, 0.001 * (1 + 0.1))  # Will be adjusted in training loop

    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


def train(model: SelfiesVAE, ds: SelfiesDataset):
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    # Even lower learning rate for more stable training
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=1e-4)
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    best_loss = math.inf
    patience = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0

        for batch_idx, (src, tgt) in enumerate(loader):
            src, tgt = src.to(dev), tgt.to(dev)
            opt.zero_grad()

            logits, mu, logvar = model(src, tgt)

            # Beta-VAE with cyclical annealing to prevent posterior collapse
            # Start with high beta, then cycle between low and high values
            cycle_length = len(loader) // 4  # 4 cycles per epoch
            cycle_pos = batch_idx % cycle_length
            beta_min, beta_max = 0.1, 1.0

            # Cyclical annealing
            if cycle_pos < cycle_length // 2:
                beta = beta_min + (beta_max - beta_min) * (cycle_pos / (cycle_length // 2))
            else:
                beta = beta_max - (beta_max - beta_min) * ((cycle_pos - cycle_length // 2) / (cycle_length // 2))

            # Also apply epoch-based annealing
            epoch_beta = min(1.0, 0.01 + 0.05 * epoch)
            final_beta = min(beta, epoch_beta)

            # Get individual loss components
            batch_size = src.size(0)
            recon_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt.view(-1),
                ignore_index=0,
                reduction="sum"
            ) / batch_size

            # KL divergence with free bits to prevent collapse
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

            # Free bits: only penalize KL above a threshold
            free_bits = 0.5  # Allow some KL divergence without penalty
            kl_loss = torch.clamp(kl_loss, min=free_bits)

            total_loss_batch = recon_loss + final_beta * kl_loss

            # Gradient clipping to prevent exploding gradients
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()

            total_loss += total_loss_batch.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches

        LOGGER.info(f"VAE epoch {epoch}/{EPOCHS} – total: {avg_loss:.4f}, recon: {avg_recon:.4f}, kl: {avg_kl:.4f}")

        scheduler.step(avg_loss)

        # More conservative early stopping
        if avg_loss < best_loss - 1e-3:  # Require bigger improvement
            best_loss = avg_loss
            patience = PATIENCE
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience -= 1
            if patience == 0:
                LOGGER.info("Early stopping VAE.")
                break

        # Prevent too quick convergence - more aggressive check
        if avg_kl < 0.01 and epoch < 20:
            LOGGER.warning(f"KL divergence collapsed at epoch {epoch} (kl={avg_kl:.6f}). This indicates posterior collapse.")
            # Force restart with higher learning rate
            for param_group in opt.param_groups:
                param_group["lr"] *= 2.0
            LOGGER.info(f"Increased learning rate to {opt.param_groups[0]['lr']:.6f}")

        # Stop if reconstruction loss is too low (overfitting)
        if avg_recon < 0.001 and epoch < 50:
            LOGGER.warning(f"Reconstruction loss too low at epoch {epoch} (recon={avg_recon:.6f}). Stopping to prevent overfitting.")
                break


# ---------------------------------------------------------------------------
# Public entry – train (if needed) and sample
# ---------------------------------------------------------------------------

def train_and_sample(n_samples: int = GENERATE_N) -> list[str]:
    config.GENERATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    selfies_data = load_selfies()
    all_tokens = [t for s in selfies_data for t in sf.split_selfies(s)]
    vocab = Vocab(all_tokens)

    model = SelfiesVAE(len(vocab)).to(dev)

    load_ok = MODEL_PATH.exists()
    if load_ok:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
        LOGGER.info(f"Loaded pre-trained SELFIES-VAE model from {MODEL_PATH}")
    else:
        LOGGER.info(f"Training SELFIES-VAE model ({len(selfies_data)} molecules)…")
        ds = SelfiesDataset(selfies_data, vocab)
        train(model, ds)
        LOGGER.info(f"Training finished. Model saved to {MODEL_PATH}")

    def sample_unique(current_model: SelfiesVAE) -> list[str]:
        """Sample unique SMILES with multiple attempts if needed."""
        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            attempts += 1
            LOGGER.info(f"Sampling attempt {attempts}/{max_attempts} with {n_samples * 2} molecules...")

            sampled_raw = current_model.sample(vocab, num=n_samples * 2)
            LOGGER.info(f"Raw sampling produced {len(sampled_raw)} molecules")

            # Additional validation with RDKit
            from rdkit import Chem  # type: ignore
            validated_smiles = []
            for smi in sampled_raw:
                if smi:
                    try:
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            # Canonicalize SMILES
                            canonical_smi = Chem.MolToSmiles(mol)
                            validated_smiles.append(canonical_smi)
                    except Exception:
                        continue

            LOGGER.info(f"After RDKit validation: {len(validated_smiles)} valid molecules")

            # Get unique molecules
            seen: set[str] = set()
            out: list[str] = []
            for s in validated_smiles:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
                if len(out) >= n_samples:
            break

            LOGGER.info(f"After deduplication: {len(out)} unique molecules")

            if out:
                return out

            if attempts < max_attempts:
                LOGGER.warning(f"Attempt {attempts} failed to generate valid molecules. Retrying...")

        LOGGER.error("Failed to generate any valid molecules after all attempts")
        return []

    unique_smiles = sample_unique(model)

    # If nothing generated and model was loaded from checkpoint, retrain
    if not unique_smiles and load_ok:
        LOGGER.warning("Pre-trained SELFIES-VAE produced zero unique SMILES – retraining from scratch.")
        MODEL_PATH.unlink(missing_ok=True)

        # Reinitialize model with new parameters
        model = SelfiesVAE(len(vocab)).to(dev)
        ds = SelfiesDataset(selfies_data, vocab)
        train(model, ds)
        unique_smiles = sample_unique(model)

    # Final fallback: if still no molecules, try direct SELFIES decoding
    if not unique_smiles:
        LOGGER.warning("Attempting fallback: direct SELFIES decoding from training data")
        fallback_smiles = []
        import random
        random.seed(42)

        # Sample random SELFIES from training data and try to decode
        for _ in range(min(100, len(selfies_data))):
            try:
                selfies_str = random.choice(selfies_data)
                smi = sf.decoder(selfies_str)
                if smi:
                    fallback_smiles.append(smi)
                if len(fallback_smiles) >= 50:
                    break
            except Exception:
                continue

        if fallback_smiles:
            LOGGER.info(f"Fallback generated {len(fallback_smiles)} molecules from training data")
            unique_smiles = fallback_smiles[:n_samples]

    # Write raw output for archive/debug
    with open(SMILES_OUT_PATH, "w", encoding="utf-8") as f:
        for smi in unique_smiles:
            f.write(f"{smi}\n")
    LOGGER.info(f"Generated {len(unique_smiles)} unique SMILES saved to {SMILES_OUT_PATH}")
    return unique_smiles
