"""Optuna-based hyperparameter optimisation for activity XGBoost model.

Usage:
    uv run python step_02_activity_prediction/optuna_tune_xgb.py --trials 50

The script tunes a handful of XGBoost parameters and saves the best model to
``config.XGB_MODEL_PATH`` as usual. Optuna study is stored in
``config.OPTUNA_STUDIES_DIR / "xgb_activity.db"`` so that subsequent runs
resume.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import optuna  # type: ignore
import polars as pl
import xgboost as xgb  # type: ignore
from rdkit import Chem, DataStructs  # type: ignore
from rdkit.Chem import Descriptors  # type: ignore
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # type: ignore

import config
from utils.logger import LOGGER

# ---------------------------------------------------------------------------
# Data utilities (same as in train_activity_model_xgb)
# ---------------------------------------------------------------------------

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


def compute_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        return None
    bitvect = _generator.GetFingerprint(mol)
    fp_arr = np.zeros((config.FP_BITS_XGB,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, fp_arr)  # type: ignore[arg-type]
    desc_vals = [func(mol) for _, func in RD_DESCS]
    feat = np.concatenate([fp_arr, np.asarray(desc_vals, dtype=np.float32)])
    return feat


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    df = pl.read_parquet(config.ACTIVITY_DATA_PROCESSED_PATH)
    feats = []
    targets = []
    for smi, ic50 in zip(df["SMILES"], df["IC50_nM"], strict=False):
        arr = compute_features(smi)
        if arr is None:
            continue
        feats.append(arr)
        targets.append(9.0 - np.log10(ic50))
    return np.vstack(feats), np.asarray(targets, dtype=np.float32)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": trial.suggest_int("max_depth", 4, 50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=150,
        evals=[(dval, "val")],
        early_stopping_rounds=25,
        verbose_eval=False,
    )
    pred = booster.predict(dval)
    rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
    return rmse


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials")
    args = parser.parse_args()

    X, y = load_dataset()
    # simple random split (activity dataset already scaffold-split earlier)
    rng = np.random.default_rng(config.RANDOM_STATE)
    permutation = rng.permutation(len(X))
    X, y = X[permutation], y[permutation]
    split = int(0.8 * len(X))
    global dtrain, dval, y_val  # pylint: disable=global-statement
    dtrain = xgb.DMatrix(X[:split], label=y[:split])
    dval = xgb.DMatrix(X[split:], label=y[split:])
    y_val = y[split:]

    study_path = config.OPTUNA_STUDIES_DIR / "xgb_activity.db"
    storage_str = f"sqlite:///{study_path}"
    LOGGER.info("Starting Optuna study at %s", storage_str)

    study = optuna.create_study(direction="minimize", study_name="xgb_activity", storage=storage_str, load_if_exists=True)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    LOGGER.info("Best RMSE %.4f with params %s", study.best_value, study.best_params)

    # Train final model with best params on full data
    best_params = {
        **config.XGB_PARAMS,  # base params
        **study.best_params,
    }
    dmat_full = xgb.DMatrix(X, label=y)
    booster = xgb.train(
        best_params,
        dmat_full,
        num_boost_round=config.XGB_NUM_BOOST_ROUND,
        evals=[(dmat_full, "train")],
        verbose_eval=False,
    )

    booster.save_model(str(config.XGB_MODEL_PATH))
    LOGGER.info("Optuna-tuned XGBoost model saved to %s", config.XGB_MODEL_PATH)

    # Save study summary
    summary_path = config.PREDICTION_RESULTS_DIR / "optuna_xgb_best.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"rmse": study.best_value, "params": study.best_params}, f, indent=2)
    LOGGER.info("Study summary written to %s", summary_path)


if __name__ == "__main__":
    main()
