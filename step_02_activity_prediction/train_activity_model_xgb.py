"""Train XGBoost regression model (GPU) for activity prediction.

Usage:
    uv run python step_02_activity_prediction/train_activity_model_xgb.py

Saves model to ``config.XGB_MODEL_PATH``.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import xgboost as xgb  # type: ignore
from rdkit import (
    Chem,  # type: ignore
    DataStructs,  # type: ignore
)
from rdkit.Chem import Descriptors  # type: ignore
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # type: ignore

import config
from utils.logger import LOGGER

# ----------------------------------------------------------------------------
# Feature engineering helpers
# ----------------------------------------------------------------------------

RD_DESCS = [
    ("MolWt", Descriptors.MolWt),  # type: ignore[attr-defined]
    ("LogP", Descriptors.MolLogP),  # type: ignore[attr-defined]
    ("TPSA", Descriptors.TPSA),  # type: ignore[attr-defined]
    ("NumHDonors", Descriptors.NumHDonors),  # type: ignore[attr-defined]
    ("NumHAcceptors", Descriptors.NumHAcceptors),  # type: ignore[attr-defined]
    ("RingCount", Descriptors.RingCount),  # type: ignore[attr-defined]
]

_generator = GetMorganGenerator(
    radius=config.FP_RADIUS,
    fpSize=config.FP_BITS_XGB,
    includeChirality=config.FP_INCLUDE_CHIRALITY,
)


def compute_features(smiles: str) -> tuple[np.ndarray, float] | None:
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        return None
    bitvect = _generator.GetFingerprint(mol)
    fp_arr = np.zeros((config.FP_BITS_XGB,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, fp_arr)  # type: ignore[arg-type]
    desc_vals = [func(mol) for _, func in RD_DESCS]
    feat = np.concatenate([fp_arr, np.asarray(desc_vals, dtype=np.float32)])
    return feat, 0.0  # placeholder second value


def load_dataset() -> pl.DataFrame:
    if not config.ACTIVITY_DATA_PROCESSED_PATH.exists():
        raise FileNotFoundError("Processed dataset missing. Run data_collection.py first.")
    return pl.read_parquet(config.ACTIVITY_DATA_PROCESSED_PATH)


# ----------------------------------------------------------------------------
# Scaffold split util
# ----------------------------------------------------------------------------

from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        return "NONE"
    scaf = MurckoScaffold.GetScaffoldForMol(mol)  # type: ignore[attr-defined]
    return Chem.MolToSmiles(scaf, isomericSmiles=True)  # type: ignore[attr-defined]


def scaffold_split(df: pl.DataFrame, test_frac: float = 0.2, seed: int = 42):
    scaffolds = {}
    for idx, smi in enumerate(df["SMILES"]):
        scaffolds.setdefault(scaffold(smi), []).append(idx)
    rng = np.random.default_rng(seed)
    keys = list(scaffolds.keys())
    rng.shuffle(keys)
    test_target = int(len(df) * test_frac)
    test_idx = []
    for k in keys:
        group = scaffolds[k]
        if len(test_idx) + len(group) <= test_target:
            test_idx.extend(group)
    mask_test = np.zeros(len(df), dtype=bool)
    mask_test[test_idx] = True
    mask_train = ~mask_test
    return mask_train, mask_test


# ----------------------------------------------------------------------------
# Training pipeline
# ----------------------------------------------------------------------------


def main() -> None:
    if getattr(config, "OPTUNA_TUNE_XGB", False):
        LOGGER.info("OPTUNA_TUNE_XGB flag is True – delegating to optuna_tune_xgb.py …")
        from step_02_activity_prediction import optuna_tune_xgb

        optuna_tune_xgb.main()
        return
    LOGGER.info("--- Training XGBoost model (GPU) ---")
    df = load_dataset()

    feats = []
    targets = []
    for smi, ic50 in zip(df["SMILES"], df["IC50_nM"], strict=False):
        # Skip zero or negative IC50 values that would cause log10 issues
        if ic50 <= 0:
            continue
        out = compute_features(smi)
        if out is None:
            continue
        fp_arr, _ = out
        feats.append(fp_arr)
        targets.append(9.0 - np.log10(ic50))

    X = np.vstack(feats)
    y = np.asarray(targets, dtype=np.float32)

    m_train, m_test = scaffold_split(df, test_frac=config.TEST_SIZE, seed=config.RANDOM_STATE)
    dtrain = xgb.DMatrix(X[m_train], label=y[m_train])
    dtest = xgb.DMatrix(X[m_test], label=y[m_test])

    booster = xgb.train(
        config.XGB_PARAMS,
        dtrain,
        num_boost_round=config.XGB_NUM_BOOST_ROUND,
        evals=[(dtest, "test")],
        early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
        verbose_eval=50,
    )

    config.XGB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(config.XGB_MODEL_PATH))
    LOGGER.info(f"XGBoost model saved to {config.XGB_MODEL_PATH}")

    preds = booster.predict(dtest)
    rmse = float(np.sqrt(np.mean((preds - y[m_test]) ** 2)))
    r2 = 1.0 - float(np.sum((preds - y[m_test]) ** 2)) / float(np.sum((y[m_test] - y[m_test].mean()) ** 2))
    metrics = {"rmse_test": rmse, "r2_test": r2, "n_train": int(m_train.sum()), "n_test": int(m_test.sum())}
    with open(config.PREDICTION_RESULTS_DIR / "metrics_xgb.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info(f"Test RMSE {rmse:.3f} | R² {r2:.3f}")
    LOGGER.info("--- XGBoost training done ---")


if __name__ == "__main__":
    main()
