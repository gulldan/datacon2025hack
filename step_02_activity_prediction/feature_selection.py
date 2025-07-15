"""Feature selection via variance and correlation filters (T29).

Reads descriptor matrix produced by ``calc_descriptors.py`` and applies an
additional correlation-based filter to remove highly collinear features.  The
resulting table is saved alongside the original one and can be consumed by
downstream models.

Output
------
``results/descriptors_filtered.parquet`` containing:
    • ``SMILES`` column with canonical SMILES strings.
    • Selected numerical descriptor columns after filtering.

The script is idempotent: if the output file already exists it exits
immediately.
"""

from __future__ import annotations

import math
import sys

import polars as pl
import polars_ds as pds

import config
from utils.logger import LOGGER

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the descriptor matrix produced by calc_descriptors.py
DESC_PATH = config.PREDICTION_RESULTS_DIR / "descriptors.parquet"

# Output with selected features
OUT_PATH = config.PREDICTION_RESULTS_DIR / "descriptors_filtered.parquet"

# Correlation threshold – if |corr| > THR the later feature is dropped
CORR_THRESHOLD = 0.95

PADel_PATH = config.PREDICTION_RESULTS_DIR / "padel_descriptors.parquet"

# ---------------------------------------------------------------------------


def greedy_correlation_filter(df_num: pl.DataFrame, threshold: float) -> list[str]:
    """Greedy selection of columns based on pair-wise Pearson correlation using *polars*.

    The algorithm preserves the *original* column order:

    • The very first numeric column is always selected.
    • For each subsequent column we compute its Pearson correlation against *each* of the
      already selected features using ``pl.corr``. If *all* absolute correlations are
      **below** ``threshold`` the column is kept.

    Any NaNs produced by ``pl.corr`` (e.g. when a column has zero variance) are treated
    as correlation 0.0 to avoid false positives and the NumPy divide-by-zero warnings
    observed with the previous pandas implementation.

    Parameters
    ----------
    df_num : polars.DataFrame
        DataFrame containing **only numeric** descriptor columns (no SMILES).
    threshold : float
        Absolute Pearson correlation cut-off. Common values: 0.9–0.95.

    Returns:
    -------
    list[str]
        Names of the selected columns, in the order they appear in *df_num*.
    """
    if df_num.width == 0:
        return []

    # cast→float64 + NaN вместо Null (так безопаснее для math.isnan)
    df = df_num.with_columns(pl.all().cast(pl.Float64).fill_null(float("nan")))

    selected: list[str] = []
    for col in df.columns:
        if not selected:  # первый признак берём всегда
            selected.append(col)
            continue

        # единственный select: корреляции col со всеми выбранными
        exprs = [pds.corr(pl.col(col), pl.col(sel)).alias(sel) for sel in selected]
        corrs = df.select(exprs).row(0)  # tuple с float | None | NaN

        # превращаем None/NaN в 0.0
        clean = [0.0 if (c is None or (isinstance(c, float) and math.isnan(c))) else c for c in corrs]

        if all(abs(c) < threshold for c in clean):
            selected.append(col)

    return selected


def main() -> None:
    if OUT_PATH.exists():
        LOGGER.info(f"Filtered descriptor file already exists: {OUT_PATH} – skipping.")
        sys.exit(0)

    if not DESC_PATH.exists():
        LOGGER.error("Descriptor matrix not found. Run calc_descriptors.py first.")
        sys.exit(1)

    LOGGER.info(f"Reading RDKit/Mordred descriptors from {DESC_PATH}")
    df = pl.read_parquet(DESC_PATH)

    # ------------------------------------------------------------------
    # Optional: merge PaDEL descriptors
    # ------------------------------------------------------------------

    if getattr(config, "USE_PADEL_DESCRIPTORS", False) and PADel_PATH.exists():
        LOGGER.info(f"Merging PaDEL descriptors from {PADel_PATH}")
        df_padel = pl.read_parquet(PADel_PATH)

        # Handle potential duplicate column names (very unlikely, but safe)
        dup_cols = set(df_padel.columns).intersection(df.columns) - {"SMILES"}
        if dup_cols:
            rename_mapping = {c: f"padel_{c}" for c in dup_cols}
            df_padel = df_padel.rename(rename_mapping)

        df = df.join(df_padel, on="SMILES", how="left")
        LOGGER.info(f"Combined descriptor matrix shape after merge: {df.shape}")

    numeric_cols = [c for c in df.columns if c != "SMILES"]
    LOGGER.info(f"Initial number of descriptor columns: {len(numeric_cols)}")

    # Run greedy correlation filter directly on a polars DataFrame (no pandas conversion)
    df_num_pl = df.select(numeric_cols)

    selected_cols = greedy_correlation_filter(df_num_pl, threshold=CORR_THRESHOLD)
    LOGGER.info(f"Kept {len(selected_cols)} columns after correlation filter (thr={CORR_THRESHOLD:.2f}).")

    # Create final DataFrame and save
    df_selected = pl.concat([df.select("SMILES"), df.select(selected_cols)], how="horizontal")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_selected.write_parquet(OUT_PATH)
    LOGGER.info(f"Filtered descriptors saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
