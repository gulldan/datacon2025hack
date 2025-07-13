"""Validation & cleaning of generated SMILES (T6).

Reads raw output ``generated_smiles_raw.txt`` produced by ``char_rnn_generator.py``,
performs:
  * RDKit validity check (again, just in case)
  * Deduplication
  * Novelty check vs training set (drop if present in processed dataset)
  * Basic diversity stats – pair-wise Tanimoto similarity (sample 500 owing to O(N²))
  * Writes valid, unique, novel structures to ``generated_molecules.parquet`` for
    downstream scoring (T7, T8).
"""
from __future__ import annotations

import random
import sys as _sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in _sys.path:
    _sys.path.insert(0, str(ROOT_DIR))

import polars as pl
from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem, DataStructs  # type: ignore

import config
from utils.logger import LOGGER

RAW_PATH = config.GENERATION_RESULTS_DIR / "generated_smiles_raw.txt"
OUT_PATH = config.GENERATION_RESULTS_DIR / "generated_molecules.parquet"


def compute_fingerprint(mol: Chem.Mol):  # type: ignore[valid-type]
    """Return Morgan fingerprint as RDKit ExplicitBitVect."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)  # type: ignore[attr-defined]


def load_training_set() -> set[str]:
    df = pl.read_parquet(config.ACTIVITY_DATA_PROCESSED_PATH)
    return set(df["SMILES"].to_list())


def main() -> None:
    if not RAW_PATH.exists():
        LOGGER.error("Raw generated file not found: %s", RAW_PATH)
        return

    smiles_raw = [s.strip() for s in RAW_PATH.read_text().splitlines() if s.strip()]
    LOGGER.info("Loaded %d raw SMILES.", len(smiles_raw))

    seen_train = load_training_set()

    valid: list[str] = []
    fps: list = []

    for smi in smiles_raw:
        if smi in seen_train:
            continue  # skip non-novel
        mol = Chem.MolFromSmiles(smi)  # type: ignore[attr-defined]
        if mol is None:
            continue
        valid.append(smi)
        fps.append(compute_fingerprint(mol))

    unique_smiles = sorted(set(valid))
    LOGGER.info("Valid & novel molecules: %d (%.1f%%)", len(unique_smiles), 100 * len(unique_smiles) / len(smiles_raw))

    # Diversity – average pairwise Tanimoto on sample
    sample_size = min(500, len(unique_smiles))
    sample_idx = random.sample(range(len(unique_smiles)), k=sample_size)
    sample_fps = [fps[i] for i in sample_idx]

    tan_sum = 0.0
    n_pairs = 0
    for i in range(sample_size):
        sims = DataStructs.BulkTanimotoSimilarity(sample_fps[i], sample_fps[i + 1 :])  # type: ignore[arg-type]
        tan_sum += sum(sims)
        n_pairs += len(sims)
    avg_tanimoto = tan_sum / n_pairs if n_pairs else 0.0
    LOGGER.info("Diversity (avg Tanimoto, sample %d) = %.3f", sample_size, avg_tanimoto)

    # Save
    out_df = pl.DataFrame({"smiles": unique_smiles})
    out_df.write_parquet(OUT_PATH)
    LOGGER.info("Validated molecules written to %s", OUT_PATH)


if __name__ == "__main__":
    main()
