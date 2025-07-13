from __future__ import annotations

"""Train activity prediction model with scaffold-based split.

This script is an alternative to ``train_activity_model.py`` using a more
realistic data split where Bemis–Murcko scaffolds are kept exclusive between
train and test.  Everything else (fingerprint generation, ElasticNet model)
remains unchanged.

Run:
    uv run python step_02_activity_prediction/train_activity_model_scaffold.py
"""

import json
from collections import defaultdict

import numpy as np
import polars as pl
from polars_ds.linear_models import ElasticNet  # type: ignore
from rdkit import Chem  # type: ignore
from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore

import config
from step_02_activity_prediction.model_utils import smiles_to_fp
from utils.logger import LOGGER

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

# Modern fingerprint helper from model_utils (uses MorganGenerator).

# (kept wrapper for potential extra options in future but delegating logic)
def morgan_fp(smiles: str, n_bits: int = 2048, radius: int = 2):
    return smiles_to_fp(smiles, n_bits=n_bits, radius=radius)


def bemis_murcko_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)  # type: ignore[attr-defined]
    return Chem.MolToSmiles(scaffold, isomericSmiles=True)  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# Dataset loading & splitting
# -----------------------------------------------------------------------------

def load_dataset() -> pl.DataFrame:
    path = config.ACTIVITY_DATA_PROCESSED_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Run data_collection.py first."
        )
    LOGGER.info("Loading processed dataset %s…", path)
    return pl.read_parquet(path)


def scaffold_split(df: pl.DataFrame, test_frac: float, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean mask arrays for train/test according to scaffold split."""
    scaffolds: dict[str, list[int]] = defaultdict(list)
    for idx, smi in enumerate(df["SMILES"], start=0):
        scaf = bemis_murcko_scaffold(smi)
        if scaf is None:
            scaf = f"NONE_{idx}"
        scaffolds[scaf].append(idx)

    rng = np.random.default_rng(seed)
    scaf_keys = np.array(list(scaffolds.keys()))
    rng.shuffle(scaf_keys)

    n_total = len(df)
    n_test_target = int(n_total * test_frac)
    test_idx: list[int] = []
    train_idx: list[int] = []

    for scaf in scaf_keys:
        group = scaffolds[scaf]
        if len(test_idx) + len(group) <= n_test_target:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    # Any remaining groups go to train
    for scaf in scaf_keys:
        if scaf not in scaffolds:
            continue
    # Convert to np bool masks
    mask_train = np.zeros(len(df), dtype=bool)
    mask_test = np.zeros(len(df), dtype=bool)
    mask_train[train_idx] = True
    mask_test[test_idx] = True
    return mask_train, mask_test


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def main() -> None:
    LOGGER.info("--- Step 2 (variant): Training with scaffold split ---")
    df = load_dataset()

    fps: list[np.ndarray] = []
    y: list[float] = []
    for smiles, ic50 in zip(df["SMILES"], df["IC50_nM"], strict=False):
        fp = smiles_to_fp(smiles)
        if fp is None:
            continue
        fps.append(fp)
        y.append(9.0 - np.log10(ic50))

    X = np.vstack(fps).astype(np.float64)
    y_vec = np.asarray(y, dtype=np.float64)

    mask_train, mask_test = scaffold_split(df, test_frac=config.TEST_SIZE, seed=config.RANDOM_STATE)
    X_train, X_test = X[mask_train], X[mask_test]
    y_train, y_test = y_vec[mask_train], y_vec[mask_test]

    LOGGER.info(f"Training samples: {len(y_train)} | Test samples (scaffold-split): {len(y_test)}")

    model = ElasticNet(l1_reg=0.001, l2_reg=0.01, has_bias=True, max_iter=5000)
    model.fit(X_train, y_train)

    def _rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def _r2(a: np.ndarray, b: np.ndarray) -> float:
        ss_res = float(np.sum((a - b) ** 2))
        ss_tot = float(np.sum((a - np.mean(a)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    metrics = {
        "r2_train": _r2(y_train, model.predict(X_train).flatten()),
        "r2_test": _r2(y_test, model.predict(X_test).flatten()),
        "rmse_train": _rmse(y_train, model.predict(X_train).flatten()),
        "rmse_test": _rmse(y_test, model.predict(X_test).flatten()),
        "n_train": int(np.sum(mask_train)),
        "n_test": int(np.sum(mask_test)),
    }

    LOGGER.info(
        "Train R² {r2_train:.3f} / RMSE {rmse_train:.3f} | Test R² {r2_test:.3f} / RMSE {rmse_test:.3f}",
        **metrics,
    )

    out_dir = config.PREDICTION_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model_out = out_dir / "activity_model_scaffold.npz"
    np.savez(model_out, coeffs=model.coeffs(), bias=model.bias())
    LOGGER.info("Scaffold-split model saved to %s", model_out)

    metrics_path = out_dir / "metrics_scaffold.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    LOGGER.info("Metrics written to %s", metrics_path)

    LOGGER.info("--- Scaffold-split training completed ---")


if __name__ == "__main__":
    main()
