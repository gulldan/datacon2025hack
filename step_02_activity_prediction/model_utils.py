"""Utility helpers to load linear fingerprint model and run predictions.

The activity model trained in ``train_activity_model.py`` is stored as an ``npz`` file
containing two arrays:
    * ``coeffs`` – weight vector of shape (n_bits,)
    * ``bias``   – scalar bias value

This module exposes helper functions to work with this artefact without relying on
pickling (``joblib``), which is not supported by the underlying ``polars_ds`` objects.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xgboost as xgb  # type: ignore
from rdkit import Chem, DataStructs  # type: ignore
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # type: ignore

import config
from utils.logger import LOGGER

# -----------------------------------------------------------------------------
# Fingerprint helpers
# -----------------------------------------------------------------------------


def smiles_to_fp(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray | None:
    """Convert SMILES to *binary* Morgan fingerprint and return as ``np.ndarray``.

    Args:
        smiles: Canonical SMILES string.
        n_bits: Fingerprint length.
        radius: Neighborhood radius.

    Returns:
        1-D ``np.ndarray`` of dtype ``float64`` (values 0/1) or ``None`` if parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        return None

    generator = GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=False)
    bitvect = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(bitvect, arr)  # type: ignore[arg-type]
    return arr


# -----------------------------------------------------------------------------
# Model loader + thin predictor class
# -----------------------------------------------------------------------------


class LinearFpModel:
    """Very small wrapper around *coeffs* + *bias* for fast dot-product prediction."""

    def __init__(self, coeffs: np.ndarray, bias: float):
        if coeffs.ndim != 1:
            raise ValueError("coeffs must be 1-D array")
        self._coeffs = coeffs.astype(np.float64)
        self._bias = float(bias)

    # ------------------------------------------------------------------
    # Public API compatible with scikit-style models used in other scripts
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """Return predictions for *X* using vectorised dot product.

        *X* can be 2-D ``np.ndarray`` (n_samples, n_bits) **or** a list of 1-D
        fingerprint arrays. Returned shape is (n_samples,).
        """
        if isinstance(X, list):
            X = np.vstack(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X @ self._coeffs + self._bias  # type: ignore[no-any-return]

    # Convenience to mirror trained ElasticNet interface used elsewhere
    def coeffs(self) -> np.ndarray:
        """Return underlying weight vector (same as ElasticNet.coeffs())."""
        return self._coeffs

    def bias(self) -> float:
        """Return scalar bias term."""
        return self._bias


# -----------------------------------------------------------------------------
# Factory / high-level helpers
# -----------------------------------------------------------------------------


def load_coefficients(path: Path | None = None) -> tuple[np.ndarray, float]:
    """Load *coeffs* and *bias* from ``npz`` artefact produced during training."""
    if path is None:
        path = config.MODEL_PATH
    data = np.load(path)
    coeffs: np.ndarray = data["coeffs"]  # type: ignore[assignment]
    bias_arr: np.ndarray = data["bias"]  # 0-D or (1,) array
    bias: float = float(bias_arr.reshape(-1)[0])
    return coeffs, bias


def load_model(path: Path | None = None) -> LinearFpModel:
    """Return prediction model – tries XGBoost first, then linear."""
    if config.XGB_MODEL_PATH.exists():
        booster = xgb.Booster()
        booster.load_model(str(config.XGB_MODEL_PATH))

        class _XGBWrapper:
            def predict(self, fp_arr):  # type: ignore[annotated-assignment]
                if isinstance(fp_arr, list):
                    fp_arr = np.vstack(fp_arr)
                if fp_arr.ndim == 1:
                    fp_arr = fp_arr.reshape(1, -1)
                dmat = xgb.DMatrix(fp_arr)
                return booster.predict(dmat)

            def coeffs(self):  # placeholder
                return np.array([])

            def bias(self):
                return 0.0

        LOGGER.info("Loaded XGBoost model from %s", config.XGB_MODEL_PATH)
        return _XGBWrapper()  # type: ignore[return-value]

    # fallback linear
    coeffs, bias = load_coefficients(path)
    LOGGER.info("Loaded activity model coeffs (%d bits) + bias from %s", len(coeffs), path or config.MODEL_PATH)
    return LinearFpModel(coeffs, bias)


# -----------------------------------------------------------------------------
# Bulk prediction helper used by CLI script(s)
# -----------------------------------------------------------------------------


def predict_smiles(smiles_iter: Iterable[str], model: LinearFpModel | None = None) -> list[float]:
    """Predict pIC50 for an *iterable* of SMILES.

    Invalid SMILES entries result in ``np.nan`` prediction.
    """
    if model is None:
        model = load_model()

    smiles_list = list(smiles_iter)
    preds: list[float] = [np.nan] * len(smiles_list)

    from rdkit.Chem import AllChem, Descriptors  # type: ignore

    # detector: if model has coeffs with length>0 -> linear; else XGB
    use_linear = hasattr(model, "coeffs") and len(model.coeffs()) > 0  # type: ignore[arg-type]

    RD_FUNCS = [
        Descriptors.MolWt,  # type: ignore[attr-defined]
        Descriptors.MolLogP,  # type: ignore[attr-defined]
        Descriptors.TPSA,  # type: ignore[attr-defined]
        Descriptors.NumHDonors,  # type: ignore[attr-defined]
        Descriptors.NumHAcceptors,  # type: ignore[attr-defined]
        Descriptors.RingCount,  # type: ignore[attr-defined]
    ]

    feat_batch: list[np.ndarray] = []
    idx_map: list[int] = []
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)  # type: ignore[attr-defined]
        if mol is None:
            continue
        fp = smiles_to_fp(smi, n_bits=1024 if not use_linear else 2048)
        if fp is None:
            continue
        if use_linear:
            arr = fp
        else:
            desc_vals = np.asarray([f(mol) for f in RD_FUNCS], dtype=np.float32)
            fp_short = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)  # type: ignore[attr-defined]
            fp_arr = np.asarray(fp_short, dtype=np.float32)
            arr = np.concatenate([fp_arr, desc_vals])
        feat_batch.append(arr)
        idx_map.append(idx)

    if feat_batch:
        X = np.vstack(feat_batch)
        batch_preds = model.predict(X).tolist()
        for i, pred_val in zip(idx_map, batch_preds, strict=False):
            preds[i] = float(pred_val)

    return preds
