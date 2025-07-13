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
    """Return ``LinearFpModel`` ready for inference."""
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

    preds: list[float] = []
    fps_batch: list[np.ndarray] = []
    idx_map: list[int] = []

    smiles_list = list(smiles_iter)
    # Build fingerprints; keep track of indices to restore order
    for idx, smi in enumerate(smiles_list):
        fp = smiles_to_fp(smi)
        if fp is None:
            preds.append(np.nan)  # placeholder
            continue
        idx_map.append(idx)
        fps_batch.append(fp)
        preds.append(0.0)  # dummy, will be replaced

    if fps_batch:
        X = np.vstack(fps_batch)
        batch_preds = model.predict(X).tolist()
        for i, pred_val in zip(idx_map, batch_preds, strict=False):
            preds[i] = float(pred_val)

    return preds
