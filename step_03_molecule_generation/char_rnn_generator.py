"""Character-level RNN generator for SMILES strings.

The script trains a small LSTM language model on canonical SMILES from
``config.ACTIVITY_DATA_PROCESSED_PATH`` and then samples novel molecules.

It is **not** a state-of-the-art model (e.g. SMILES-VAE, REINVENT) but serves as
an educational baseline that
1. demonstrates end-to-end training with PyTorch,
2. produces 1-5k candidate molecules for downstream filtering (T6–T8).

Outputs
-------
* ``config.GENERATION_RESULTS_DIR / "generated_smiles_raw.txt"`` – plain SMILES one per line (may contain invalid / duplicate entries).
* ``config.GENERATION_RESULTS_DIR / "char_rnn.pt"`` – trained model weights.

The script is idempotent – if model + output already exist it will skip training
and just sample.
"""
from __future__ import annotations

import math
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
import sys as _sys

if str(ROOT_DIR) not in _sys.path:
    _sys.path.insert(0, str(ROOT_DIR))
import random
import sys

import numpy as np
import torch
from rdkit import Chem  # type: ignore
from torch import nn
from torch.utils.data import DataLoader, Dataset

import config
from utils.logger import LOGGER

# -----------------------------------------------------------------------------
# Hyper-parameters (could be exposed via argparse for fine control)
# -----------------------------------------------------------------------------

EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 256  # chars per batch sequence
SEQ_LEN = 120     # truncate / pad SMILES to this length
EPOCHS = 10       # quick demo – increase for quality
GENERATE_N = 2000 # number of SMILES to sample after training
LEARNING_RATE = 1e-3
PATIENCE = 3      # early-stopping patience (epochs)

MODEL_PATH = config.GENERATION_RESULTS_DIR / "char_rnn.pt"
SMILES_OUT_PATH = config.GENERATION_RESULTS_DIR / "generated_smiles_raw.txt"

SEED = config.RANDOM_STATE
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_smiles() -> list[str]:
    """Load canonical SMILES from processed activity dataset."""
    if not config.ACTIVITY_DATA_PROCESSED_PATH.exists():
        LOGGER.error("Processed dataset not found: %s", config.ACTIVITY_DATA_PROCESSED_PATH)
        sys.exit(1)
    import polars as pl

    df = pl.read_parquet(config.ACTIVITY_DATA_PROCESSED_PATH)
    smiles_list: list[str] = df["SMILES"].unique().to_list()  # type: ignore[no-any-return]
    return list(smiles_list)


# -----------------------------------------------------------------------------
# Dataset / Vocabulary helpers
# -----------------------------------------------------------------------------


class Vocab:
    def __init__(self, tokens: list[str]):
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(tokens)}
        self.itos: list[str] = tokens

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, idx: list[int]) -> str:
        return "".join(self.itos[i] for i in idx)

    def __len__(self) -> int:
        return len(self.itos)


class SmilesDataset(Dataset):
    def __init__(self, smiles: list[str], vocab: Vocab):
        self.vocab = vocab
        self.data = []
        # prepend <bos>, append <eos>
        bos, eos, pad = "^", "$", "_"
        for smi in smiles:
            seq = bos + smi + eos
            seq = seq[: SEQ_LEN]
            seq += pad * max(0, SEQ_LEN - len(seq))  # pad to fixed length
            self.data.append(vocab.encode(seq))
        self.data = torch.tensor(self.data, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):  # type: ignore[override]
        x = self.data[idx]
        return x[:-1], x[1:]  # input, target shifted by 1


# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------


class CharRNN(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_DIM)
        self.lstm = nn.LSTM(
            EMBED_DIM,
            HIDDEN_DIM,
            NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
        )
        self.fc = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, x, hidden=None):  # type: ignore[override]
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden

    def sample(self, vocab: Vocab, max_len: int = 120, device: str = "cpu") -> str:
        bos_idx = vocab.stoi["^"]
        eos_idx = vocab.stoi["$"]
        pad_idx = vocab.stoi["_"]
        idx = bos_idx
        hidden = None
        seq: list[int] = []
        for _ in range(max_len):
            inp = torch.tensor([[idx]], dtype=torch.long, device=device)
            logits, hidden = self.forward(inp, hidden)
            probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze(0)
            idx = torch.multinomial(probs, num_samples=1).item()
            if idx in (eos_idx, pad_idx):
                break
            seq.append(idx)  # type: ignore[arg-type]
        return vocab.decode(seq)


# -----------------------------------------------------------------------------
# Training + sampling logic
# -----------------------------------------------------------------------------

def train_model(model: CharRNN, ds: SmilesDataset, vocab: Vocab, device: str = "cpu") -> None:
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = math.inf
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, len(vocab)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(ds)
        LOGGER.info("Epoch %d/%d – loss %.4f", epoch, EPOCHS, avg_loss)

        # Early stopping
        if avg_loss + 1e-4 < best_loss:
            best_loss = avg_loss
            patience_left = PATIENCE
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_left -= 1
            if patience_left == 0:
                LOGGER.info("Early stopping triggered.")
                break


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Using device: %s", device)

    if MODEL_PATH.exists() and SMILES_OUT_PATH.exists():
        LOGGER.info("Model + generated SMILES already exist. Skipping training.")
    else:
        smiles_list = load_smiles()
        # Build vocabulary
        special = ["^", "_", "$"]  # bos, pad, eos
        charset = sorted({c for smi in smiles_list for c in smi}.union(special))
        vocab = Vocab(charset)
        LOGGER.info("Vocab size: %d", len(vocab))

        ds = SmilesDataset(smiles_list, vocab)
        model = CharRNN(len(vocab)).to(device)
        train_model(model, ds, vocab, device)
        LOGGER.info("Training finished, model saved to %s", MODEL_PATH)

    # Load best model + vocab for sampling
    smiles_list = load_smiles()
    special = ["^", "_", "$"]
    charset = sorted({c for smi in smiles_list for c in smi}.union(special))
    vocab = Vocab(charset)
    model = CharRNN(len(vocab))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    LOGGER.info("Sampling %d SMILES…", GENERATE_N)
    generated: list[str] = []
    while len(generated) < GENERATE_N:
        smi = model.sample(vocab)
        # RDKit validity check
        mol = Chem.MolFromSmiles(smi)  # type: ignore[attr-defined]
        if mol is None:
            continue
        generated.append(Chem.MolToSmiles(mol))  # type: ignore[attr-defined] # canonicalise
    # Remove duplicates + ones present in training set
    unique_new = sorted({s for s in generated if s not in smiles_list})
    LOGGER.info("Obtained %d unique novel molecules.", len(unique_new))

    # Persist output
    SMILES_OUT_PATH.write_text("\n".join(unique_new))
    LOGGER.info("Generated SMILES written to %s", SMILES_OUT_PATH)


if __name__ == "__main__":
    main()
