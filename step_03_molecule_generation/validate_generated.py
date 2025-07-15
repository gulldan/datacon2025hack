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
from rdkit import (
    Chem,  # type: ignore
    DataStructs,  # type: ignore
)
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # type: ignore

import config
from utils.logger import LOGGER

RAW_PATH = config.GENERATION_RESULTS_DIR / "generated_smiles_raw.txt"
OUT_PATH = config.GENERATION_RESULTS_DIR / "generated_molecules.parquet"


_fp_gen = GetMorganGenerator(
    radius=config.FP_RADIUS,
    fpSize=config.FP_BITS_LINEAR,
    includeChirality=config.FP_INCLUDE_CHIRALITY,
)


def fingerprint(mol):
    """Return RDKit fingerprint object for Tanimoto similarity calculations."""
    return _fp_gen.GetFingerprint(mol)


def _canonical(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)  # type: ignore[attr-defined]
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True)  # type: ignore[attr-defined]


def load_training_set() -> tuple[list[str], list]:
    """Return canonical SMILES and fingerprints of training set."""
    df = pl.read_parquet(config.ACTIVITY_DATA_PROCESSED_PATH)
    smiles_train = []
    fps_train = []
    for s in df["SMILES"]:
        can = _canonical(s)
        if not can:
            continue
        smiles_train.append(can)
        mol = Chem.MolFromSmiles(can)  # type: ignore[attr-defined]
        fps_train.append(fingerprint(mol))
    return smiles_train, fps_train


def main() -> None:
    if not RAW_PATH.exists():
        LOGGER.error(f"Raw generated file not found: {RAW_PATH}")
        return

    smiles_raw = [s.strip() for s in RAW_PATH.read_text().splitlines() if s.strip()]

    if not smiles_raw:
        LOGGER.warning("Raw SMILES list is empty – nothing to validate. Skipping further checks.")
        return

    LOGGER.info(f"Loaded {len(smiles_raw)} raw SMILES.")

    train_smiles, train_fps = load_training_set()

    valid: list[str] = []
    fps: list = []

    NOVEL_TANIMOTO_THRESH = 0.9
    for smi in smiles_raw:
        can = _canonical(smi)
        if not can:
            continue
        mol = Chem.MolFromSmiles(can)  # type: ignore[attr-defined]
        if mol is None:
            continue
        fp = fingerprint(mol)

        # novelty via Tanimoto < 0.9 to any training compound
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)  # type: ignore[arg-type]
        if sims and max(sims) >= NOVEL_TANIMOTO_THRESH:
            continue  # too similar to known compound

        valid.append(can)
        fps.append(fp)

    unique_smiles = sorted(set(valid))
    valid_pct = 100 * len(unique_smiles) / len(smiles_raw) if smiles_raw else 0.0
    LOGGER.info(f"Valid & novel molecules: {len(unique_smiles)} ({valid_pct:.1f}%)")

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
    LOGGER.info(f"Diversity (avg Tanimoto, sample {sample_size}) = {avg_tanimoto:.3f}")

    # Preserve previously computed scores if they exist
    if OUT_PATH.exists():
        try:
            prev_df = pl.read_parquet(OUT_PATH)
            if "smiles" in prev_df.columns and len(prev_df) > 0:
                prev_df = prev_df.filter(pl.col("smiles").is_in(unique_smiles))
                out_df = prev_df
            else:
                out_df = pl.DataFrame({"smiles": unique_smiles})
        except Exception:
            out_df = pl.DataFrame({"smiles": unique_smiles})
    else:
        out_df = pl.DataFrame({"smiles": unique_smiles})

    out_df.write_parquet(OUT_PATH)
    LOGGER.info(f"Validated molecules written to {OUT_PATH}")


if __name__ == "__main__":
    main()
