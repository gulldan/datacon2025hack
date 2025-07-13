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
EMBED_DIM = 196
HIDDEN_DIM = 392
LATENT_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
MAX_LEN = 120  # SELFIES tokens (covers >99 % dataset)
EPOCHS = config.MAX_VAE_EPOCHS
LEARNING_RATE = 1e-3
MODEL_PATH = config.GENERATION_RESULTS_DIR / "selfies_vae.pt"
SMILES_OUT_PATH = config.GENERATION_RESULTS_DIR / "generated_smiles_raw.txt"
GENERATE_N = 2000
PATIENCE = 4

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
        for i in idx:
            tok = self.itos[i]
            if tok in (self.eos, self.pad, self.bos):
                break
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
    AUG_PER_MOL = 10  # количество рандомных перестановок SMILES
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
        self.eval()
        with torch.no_grad():
            z = torch.randn(num, LATENT_DIM, device=dev)
            bos_idx = vocab.stoi[vocab.bos]
            pad_idx = vocab.stoi[vocab.pad]

            seqs = torch.full((num, MAX_LEN), pad_idx, dtype=torch.long, device=dev)
            seqs[:, 0] = bos_idx
            for t in range(1, MAX_LEN):
                logits = self.dec(z, seqs[:, :t])[:, -1, :]
                prob = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(prob, 1).squeeze(-1)
                seqs[:, t] = next_tok
            smiles_out: list[str] = []
            for row in seqs:
                tokens = vocab.decode(row.tolist())
                selfies_str = "".join(tokens)
                try:
                    smi = sf.decoder(selfies_str)
                    smiles_out.append(smi)
                except sf.DecoderError:
                    continue
            return smiles_out


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def loss_fn(logits, targets, mu, logvar):
    recon_loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.1 * kld


def train(model: SelfiesVAE, ds: SelfiesDataset):
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best = math.inf
    patience = PATIENCE
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot = 0.0
        for src, tgt in loader:
            src, tgt = src.to(dev), tgt.to(dev)
            opt.zero_grad()
            logits, mu, logvar = model(src, tgt)
            loss = loss_fn(logits, tgt, mu, logvar)
            loss.backward()
            opt.step()
            tot += loss.item() * src.size(0)
        avg = tot / len(ds)
        LOGGER.info("VAE epoch %d/%d – loss %.4f", epoch, EPOCHS, avg)
        if avg < best - 1e-4:
            best = avg
            patience = PATIENCE
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience -= 1
            if patience == 0:
                LOGGER.info("Early stopping VAE.")
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

    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
        LOGGER.info("Loaded pre-trained SELFIES-VAE model from %s", MODEL_PATH)
    else:
        LOGGER.info("Training SELFIES-VAE model (%d molecules)…", len(selfies_data))
        ds = SelfiesDataset(selfies_data, vocab)
        train(model, ds)
        LOGGER.info("Training finished. Model saved to %s", MODEL_PATH)

    sampled = model.sample(vocab, num=n_samples * 4)  # oversample – later filter unique/valid
    unique_smiles = []
    seen = set()
    for smi in sampled:
        if smi and smi not in seen:
            seen.add(smi)
            unique_smiles.append(smi)
        if len(unique_smiles) >= n_samples:
            break

    # Write raw output for archive/debug
    with open(SMILES_OUT_PATH, "w", encoding="utf-8") as f:
        for smi in unique_smiles:
            f.write(f"{smi}\n")
    LOGGER.info("Generated %d unique SMILES saved to %s", len(unique_smiles), SMILES_OUT_PATH)
    return unique_smiles
