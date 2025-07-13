"""Optuna hyperparameter tuning for SELFIES-VAE.

Searches over architecture & optimisation hyper-parameters and stores best
configuration in ``generation_results/optuna_vae_best.json``.

This script is automatically invoked from ``selfies_vae_generator.py`` when
``config.OPTUNA_TUNE_VAE`` is True, but can also be run standalone:

    uv run python step_03_molecule_generation/optuna_tune_vae.py --trials 30
"""
from __future__ import annotations

import argparse
import json

import optuna  # type: ignore
import torch
from torch import nn
from torch.utils.data import DataLoader

import config

# Import after config so constants exist
from step_03_molecule_generation.selfies_vae_generator import (
    SelfiesDataset,
    SelfiesVAE,
    Vocab,
    load_selfies,
)
from utils.logger import LOGGER

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Objective helper
# ---------------------------------------------------------------------------

def build_model(vocab_size: int, trial: optuna.Trial):
    import step_03_molecule_generation.selfies_vae_generator as gmod

    gmod.EMBED_DIM = trial.suggest_categorical("embed_dim", [128, 160, 192, 224])
    gmod.HIDDEN_DIM = trial.suggest_categorical("hidden_dim", [256, 320, 384, 448])
    gmod.LATENT_DIM = trial.suggest_categorical("latent_dim", [64, 96, 128, 160])
    gmod.DROPOUT = trial.suggest_float("dropout", 0.1, 0.4)
    gmod.LEARNING_RATE = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    gmod.BATCH_SIZE = trial.suggest_categorical("batch_size", [32, 64, 96])
    return SelfiesVAE(vocab_size).to(DEV)


def objective(trial: optuna.Trial):
    selfies_list = load_selfies()
    vocab = Vocab([t for s in selfies_list for t in s])
    ds = SelfiesDataset(selfies_list, vocab)

    # 80/20 split
    n_train = int(0.8 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    model = build_model(len(vocab), trial)
    opt = torch.optim.Adam(model.parameters(), lr=model.LEARNING_RATE if hasattr(model, "LEARNING_RATE") else 1e-3)  # type: ignore[arg-type]

    bs = getattr(model, "BATCH_SIZE", 64)  # type: ignore[attr-defined]
    loader_train = DataLoader(train_ds, batch_size=int(bs), shuffle=True)
    loader_val = DataLoader(val_ds, batch_size=128)

    def loss_fn(logits, targets, mu, logvar):  # copy from generator
        recon_loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 0.1 * kld

    # Train few epochs
    model.train()
    for epoch in range(3):
        for src, tgt in loader_train:
            src, tgt = src.to(DEV), tgt.to(DEV)
            opt.zero_grad()
            logits, mu, logvar = model(src, tgt)
            loss = loss_fn(logits, tgt, mu, logvar)
            loss.backward()
            opt.step()

    # Validation loss
    model.eval()
    val_tot = 0.0
    with torch.no_grad():
        for src, tgt in loader_val:
            src, tgt = src.to(DEV), tgt.to(DEV)
            logits, mu, logvar = model(src, tgt)
            val_tot += loss_fn(logits, tgt, mu, logvar).item() * src.size(0)
    return val_tot / len(val_ds)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    study_path = config.OPTUNA_STUDIES_DIR / "vae_generator.db"
    storage = f"sqlite:///{study_path}"

    study = optuna.create_study(direction="minimize", study_name="vae_generator", storage=storage, load_if_exists=True)
    LOGGER.info("Starting VAE Optuna study (%d trials)…", args.trials)
    study.optimize(objective, n_trials=args.trials)

    LOGGER.info("Best loss %.4f with params %s", study.best_value, study.best_params)

    out_json = config.GENERATION_RESULTS_DIR / "optuna_vae_best.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"loss": study.best_value, "params": study.best_params}, f, indent=2)
    LOGGER.info("Saved best params to %s", out_json)


if __name__ == "__main__":
    main()
