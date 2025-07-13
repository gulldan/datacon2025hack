from __future__ import annotations

"""Train activity prediction model for DYRK1A.

Steps:
1. Load processed dataset produced by ``data_collection.py``.
2. Compute pIC50 target (9 - log10(IC50_nM)).
3. Generate RDKit Morgan fingerprints for each molecule.
4. Split into train/test using simple random split (80/20, reproducible).
5. Train ElasticNet regressor from ``polars_ds``.
6. Evaluate (R², RMSE) on train & test.
7. Save fitted model to ``config.MODEL_PATH``.
8. Write metrics to ``results/metrics.json`` and plot feature importances.

Requirements: RDKit, numpy, polars, polars_ds, joblib, plotly.
"""

import json
from pathlib import Path

import numpy as np
import plotly.express as px  # type: ignore
import polars as pl
from polars_ds.linear_models import ElasticNet  # type: ignore

import config
from step_02_activity_prediction.model_utils import smiles_to_fp
from utils.logger import LOGGER

# We now rely on modern RDKit MorganGenerator implementation provided via
# ``model_utils.smiles_to_fp`` to avoid deprecation warnings.


# -----------------------------------------------------------------------------
# Training pipeline
# -----------------------------------------------------------------------------

def load_dataset(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Run step_02_activity_prediction/data_collection.py first."
        )
    LOGGER.info("Loading processed dataset %s…", path)
    return pl.read_parquet(path)


def compute_target(df: pl.DataFrame) -> pl.Series:
    """Compute pIC50 from IC50_nM column."""
    return 9.0 - np.log10(df["IC50_nM"].to_numpy())  # type: ignore[arg-type]


def build_feature_matrix(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Generate feature matrix X and target vector y."""
    fingerprints: list[np.ndarray] = []
    targets: list[float] = []

    for smiles, ic50_nm in zip(df["SMILES"], df["IC50_nM"], strict=False):
        fp = smiles_to_fp(smiles)
        if fp is None:
            continue
        fingerprints.append(fp)
        targets.append(9.0 - np.log10(ic50_nm))

    X = np.vstack(fingerprints).astype(np.float64)  # shape (n_samples, n_bits)
    y = np.asarray(targets, dtype=np.float64)
    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> tuple[ElasticNet, dict[str, float]]:
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(idx) * (1.0 - config.TEST_SIZE))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = ElasticNet(l1_reg=0.001, l2_reg=0.01, has_bias=True, max_iter=5000)
    LOGGER.info(f"Training ElasticNet model ({len(X_train)} train samples)…")
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
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }
    return model, metrics


def plot_feature_importance(model: ElasticNet, out_path: Path):
    importances = np.abs(model.coeffs())
    feature_names = [f"Bit_{i}" for i in range(len(importances))]
    top_idx = np.argsort(importances)[-20:][::-1]
    fig = px.bar(
        x=importances[top_idx],
        y=[feature_names[i] for i in top_idx],
        orientation="h",
        title="Top-20 important fingerprint bits (ElasticNet coef abs)",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.write_html(out_path)
    LOGGER.info("Feature-importance plot saved to %s", out_path)


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main() -> None:
    LOGGER.info("--- Step 2: Vectorisation + model training ---")

    df = load_dataset(config.ACTIVITY_DATA_PROCESSED_PATH)
    X, y = build_feature_matrix(df)

    model, metrics = train_model(X, y)
    LOGGER.info(
        "Train R² {r2_train:.3f} / RMSE {rmse_train:.3f} | Test R² {r2_test:.3f} / RMSE {rmse_test:.3f}",
        **metrics,
    )

    # Save artefacts
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(config.MODEL_PATH, coeffs=model.coeffs(), bias=model.bias())
    LOGGER.info("Model coefficients saved to %s (npz)", config.MODEL_PATH)

    metrics_path = config.PREDICTION_RESULTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    LOGGER.info("Metrics written to %s", metrics_path)

    importance_plot_path = config.FEATURE_IMPORTANCE_PATH
    plot_feature_importance(model, importance_plot_path)

    LOGGER.info("--- Step 2 completed ---")


if __name__ == "__main__":
    main()
