"""Transformer-based VAE for molecular generation with improved architecture.
Based on recent research in molecular generation and attention mechanisms.
Implements Cyclical Annealing Schedule to fix KL vanishing problem.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import selfies as sf  # type: ignore
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import config
from utils.logger import LOGGER

# Device configuration
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration from config.py
BATCH_SIZE = config.VAE_BATCH_SIZE
MAX_LEN = config.VAE_MAX_LEN
EMBED_DIM = config.VAE_EMBED_DIM
HIDDEN_DIM = config.VAE_HIDDEN_DIM
LATENT_DIM = config.VAE_LATENT_DIM
LEARNING_RATE = config.VAE_LEARNING_RATE
EPOCHS = config.MAX_VAE_EPOCHS
PATIENCE = config.VAE_PATIENCE
GENERATE_N = config.VAE_GENERATE_N

# Paths
MODEL_PATH = config.GENERATION_RESULTS_DIR / "transformer_vae.pt"
SMILES_OUT_PATH = config.GENERATION_RESULTS_DIR / "generated_smiles_raw.txt"


def cyclical_annealing(step: int, total_steps: int, n_cycles: int = 4, ratio: float = 0.5, max_beta: float = 1.0) -> float:
    """Cyclical annealing schedule for β parameter.
    
    Based on: "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing"
    https://arxiv.org/abs/1903.10145
    
    Args:
        step: Current training step
        total_steps: Total number of training steps
        n_cycles: Number of cycles
        ratio: Proportion used to increase β (0 < ratio < 1)
        max_beta: Maximum β value
    
    Returns:
        Current β value
    """
    period = total_steps / n_cycles
    cycle_step = step % period
    tau = cycle_step / period

    if tau <= ratio:
        # Linear increase from 0 to max_beta
        beta = max_beta * (tau / ratio)
    else:
        # Keep at max_beta
        beta = max_beta

    return beta


def monotonic_annealing(step: int, total_steps: int, max_beta: float = 1.0, warmup_steps: int = 1000) -> float:
    """Monotonic annealing schedule for β parameter.
    
    Args:
        step: Current training step
        total_steps: Total number of training steps
        max_beta: Maximum β value
        warmup_steps: Number of warmup steps
    
    Returns:
        Current β value
    """
    if step < warmup_steps:
        return 0.0
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    return max_beta * progress


def logistic_annealing(step: int, total_steps: int, max_beta: float = 1.0, k: float = 0.0025) -> float:
    """Logistic annealing schedule for β parameter.
    
    Args:
        step: Current training step
        total_steps: Total number of training steps
        max_beta: Maximum β value
        k: Steepness parameter
    
    Returns:
        Current β value
    """
    x0 = total_steps * 0.25  # Midpoint
    return max_beta / (1 + np.exp(-k * (step - x0)))


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe shape: [max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        # pe is [max_len, d_model], we need [batch_size, seq_len, d_model]
        pe = self.pe[:seq_len, :].unsqueeze(0)  # [1, seq_len, d_model]
        return x + pe.expand(x.size(0), -1, -1)


class TransformerEncoder(nn.Module):
    """Transformer encoder for molecular sequences."""

    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, MAX_LEN)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # VAE components
        self.mu_head = nn.Linear(d_model, LATENT_DIM)
        self.logvar_head = nn.Linear(d_model, LATENT_DIM)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch_size, seq_len]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Transformer encoding
        encoded = self.transformer(x, src_key_padding_mask=mask)

        # Global pooling (mean over sequence)
        if mask is not None:
            # Mask out padding tokens
            mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
            encoded = encoded.masked_fill(mask_expanded, 0)
            lengths = (~mask).sum(dim=1, keepdim=True).float()
            pooled = encoded.sum(dim=1) / lengths
        else:
            pooled = encoded.mean(dim=1)

        # VAE parameters
        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled)

        return mu, logvar


class TransformerDecoder(nn.Module):
    """Transformer decoder for molecular sequences."""

    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, MAX_LEN)

        # Latent to initial hidden state
        self.latent_to_hidden = nn.Linear(LATENT_DIM, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt: torch.Tensor, latent: torch.Tensor, tgt_mask: torch.Tensor | None = None) -> torch.Tensor:
        # tgt: [batch_size, seq_len]
        # latent: [batch_size, latent_dim]

        batch_size, seq_len = tgt.shape

        # Embed target sequence
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoder(tgt_embedded)

        # Create memory from latent
        memory = self.latent_to_hidden(latent).unsqueeze(1)  # [batch_size, 1, d_model]

        # Generate causal mask for target
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(tgt.device)

        # Transformer decoding
        output = self.transformer(
            tgt_embedded,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_mask
        )

        # Project to vocabulary
        logits = self.output_proj(output)

        return logits


class TransformerVAE(nn.Module):
    """Transformer-based VAE for molecular generation."""

    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = TransformerEncoder(vocab_size, d_model, nhead, num_layers)
        self.decoder = TransformerDecoder(vocab_size, d_model, nhead, num_layers)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor | None = None,
                tgt_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode
        mu, logvar = self.encoder(src, src_mask)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        logits = self.decoder(tgt, z, tgt_mask)

        return logits, mu, logvar

    def sample(self, vocab: Any, num: int = 100, temperature: float = 1.0, top_k: int = 50) -> list[str]:
        """Generate molecules using nucleus sampling."""
        self.eval()
        dev = next(self.parameters()).device

        # Process in smaller batches to avoid memory issues
        batch_size = min(100, num)
        all_smiles = []

        with torch.no_grad():
            for start_idx in range(0, num, batch_size):
                end_idx = min(start_idx + batch_size, num)
                current_batch_size = end_idx - start_idx

                # Sample from prior
                z = torch.randn(current_batch_size, LATENT_DIM, device=dev)

                # Initialize sequences
                bos_idx = vocab.stoi[vocab.bos]
                eos_idx = vocab.stoi.get(vocab.eos, vocab.stoi[vocab.pad])
                pad_idx = vocab.stoi[vocab.pad]

                sequences = torch.full((current_batch_size, MAX_LEN), pad_idx, dtype=torch.long, device=dev)
                sequences[:, 0] = bos_idx

                # Generate sequences
                for t in range(1, MAX_LEN):
                    # Get logits for current position
                    logits = self.decoder(sequences[:, :t], z)[:, -1, :]  # [batch_size, vocab_size]

                    # Apply temperature
                    logits = logits / temperature

                    # Top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits = torch.full_like(logits, -float("inf"))
                        logits.scatter_(-1, top_k_indices, top_k_logits)

                    # Sample next token
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).squeeze(-1)

                    sequences[:, t] = next_token

                    # Stop if all sequences have generated EOS
                    if (next_token == eos_idx).all():
                        break

                # Convert to SMILES
                for seq in sequences:
                    tokens = vocab.decode(seq.tolist())
                    selfies_str = "".join(tokens)

                    if not selfies_str or selfies_str == vocab.bos:
                        continue

                    try:
                        smi = sf.decoder(selfies_str)
                        if smi and len(smi) > 3:
                            all_smiles.append(smi)
                    except Exception:
                        continue

            return all_smiles


def improved_loss_fn(logits: torch.Tensor, targets: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
                    beta: float = 1.0, free_bits: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Improved loss function with KL annealing and free bits."""
    batch_size = targets.size(0)

    # Reshape tensors properly
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)

    # Reconstruction loss
    recon_loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=0,
        reduction="sum"
    ) / batch_size

    # KL divergence loss with free bits
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    kl_loss = torch.clamp(kl_loss, min=free_bits)

    # Total loss with beta weighting
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def train_transformer_vae(model: TransformerVAE, dataloader: DataLoader, num_epochs: int = 100, annealing_type: str = "cyclical") -> None:
    """Train the Transformer VAE with cyclical annealing schedule to fix KL vanishing.
    
    Args:
        model: TransformerVAE model
        dataloader: Training data loader
        num_epochs: Number of training epochs
        annealing_type: Type of annealing ("cyclical", "monotonic", "logistic")
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_loss = float("inf")
    patience_counter = 0

    # Calculate total steps for annealing
    total_steps = num_epochs * len(dataloader)
    step_count = 0

    # Track metrics for early stopping
    best_reconstruction = float("inf")
    best_kl = 0.0
    epochs_without_improvement = 0

    LOGGER.info(f"Starting training with {annealing_type} annealing schedule...")
    LOGGER.info(f"Total steps: {total_steps}, Batches per epoch: {len(dataloader)}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        epoch_start_step = step_count

        for batch_idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(dev), tgt.to(dev)

            # Calculate beta using chosen annealing schedule
            if annealing_type == "cyclical":
                beta = cyclical_annealing(step_count, total_steps, n_cycles=config.VAE_ANNEALING_CYCLES,
                                        ratio=config.VAE_ANNEALING_RATIO, max_beta=config.VAE_MAX_BETA)
            elif annealing_type == "monotonic":
                beta = monotonic_annealing(step_count, total_steps, max_beta=config.VAE_MAX_BETA,
                                         warmup_steps=total_steps//10)
            elif annealing_type == "logistic":
                beta = logistic_annealing(step_count, total_steps, max_beta=config.VAE_MAX_BETA, k=0.0025)
            else:
                beta = 1.0  # Default constant beta

            # Create masks
            src_mask = (src == 0)  # Padding mask
            tgt_mask = (tgt == 0)  # Padding mask

            optimizer.zero_grad()

            # Forward pass
            logits, mu, logvar = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1])

            # Compute loss with current beta
            loss, recon_loss, kl_loss = improved_loss_fn(
                logits, tgt[:, 1:], mu, logvar, beta, free_bits=0.0
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

            step_count += 1

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)

        # Calculate average beta for this epoch
        epoch_beta = 0
        for i in range(len(dataloader)):
            step = epoch_start_step + i
            if annealing_type == "cyclical":
                epoch_beta += cyclical_annealing(step, total_steps, n_cycles=config.VAE_ANNEALING_CYCLES,
                                               ratio=config.VAE_ANNEALING_RATIO, max_beta=config.VAE_MAX_BETA)
            elif annealing_type == "monotonic":
                epoch_beta += monotonic_annealing(step, total_steps, max_beta=config.VAE_MAX_BETA,
                                                warmup_steps=total_steps//10)
            elif annealing_type == "logistic":
                epoch_beta += logistic_annealing(step, total_steps, max_beta=config.VAE_MAX_BETA, k=0.0025)
            else:
                epoch_beta += 1.0
        epoch_beta /= len(dataloader)

        LOGGER.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, Beta: {epoch_beta:.4f}")

        scheduler.step()

        # Improved early stopping based on reconstruction loss and KL divergence
        improvement = False

        # Check if reconstruction improved
        if avg_recon < best_reconstruction:
            best_reconstruction = avg_recon
            improvement = True

        # Check if KL divergence is healthy (not collapsed)
        if avg_kl > 0.01:  # Minimum threshold to avoid collapse
            if avg_kl > best_kl:
                best_kl = avg_kl
                improvement = True

        # Save model if overall loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)
            LOGGER.info(f"Model saved with loss: {best_loss:.4f}")

        # Early stopping logic
        if improvement:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Stop if KL completely collapsed
        if avg_kl < 0.005:  # Увеличиваем порог для более здорового KL
            LOGGER.warning(f"KL divergence collapsed to {avg_kl:.6f}. Stopping training.")
            break

        # Stop if no improvement for patience epochs
        if epochs_without_improvement >= PATIENCE:
            LOGGER.info(f"No improvement for {PATIENCE} epochs. Early stopping triggered.")
            break


# Integration function for existing pipeline
def train_and_sample_transformer(n_samples: int = GENERATE_N) -> list[str]:
    """Train and sample from Transformer VAE."""
    from step_03_molecule_generation.selfies_vae_generator import SelfiesDataset, Vocab, load_selfies

    # Load data
    selfies_data = load_selfies()
    vocab = Vocab([tok for selfies in selfies_data for tok in sf.split_selfies(selfies)])

    # Create model
    model = TransformerVAE(len(vocab)).to(dev)

    # Check if model exists
    if MODEL_PATH.exists():
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
            LOGGER.info(f"Loaded pre-trained Transformer VAE from {MODEL_PATH}")
        except Exception as e:
            LOGGER.warning(f"Failed to load model: {e}. Training from scratch.")
            MODEL_PATH.unlink(missing_ok=True)

    if not MODEL_PATH.exists():
        LOGGER.info(f"Training Transformer VAE on {len(selfies_data)} molecules...")
        dataset = SelfiesDataset(selfies_data, vocab)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        train_transformer_vae(model, dataloader, EPOCHS, annealing_type=config.VAE_ANNEALING_TYPE)
        LOGGER.info(f"Training finished. Model saved to {MODEL_PATH}")

    # Sample molecules
    LOGGER.info(f"Sampling {n_samples} molecules...")
    unique_smiles = model.sample(vocab, num=n_samples, temperature=0.9, top_k=50)

    # Save results
    with open(SMILES_OUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(f"{smi}\n" for smi in unique_smiles)

    LOGGER.info(f"Generated {len(unique_smiles)} unique SMILES saved to {SMILES_OUT_PATH}")
    return unique_smiles
