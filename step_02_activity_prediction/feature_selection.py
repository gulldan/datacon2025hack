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

import sys

import pandas as pd  # used only for correlation computation (faster than polars)
import polars as pl

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

# ---------------------------------------------------------------------------


def greedy_correlation_filter(df_num: pd.DataFrame, threshold: float) -> list[str]:
    """Greedy selection of columns based on pair-wise Pearson correlation.

    Iterates over columns in their original order, adding a column if its
    correlation with all *already selected* columns is below ``threshold``.

    Parameters
    ----------
    df_num : pandas.DataFrame
        DataFrame with only numeric columns (no SMILES).
    threshold : float
        Absolute Pearson correlation cutoff.

    Returns:
    -------
    List[str]
        Names of selected columns.
    """
    selected: list[str] = []
    for col in df_num.columns:
        if not selected:
            selected.append(col)
            continue

        # Compute correlation vs already kept columns; break early if any high corr
        corrs = df_num[selected].corrwith(df_num[col]).abs()
        if (corrs < threshold).all():
            selected.append(col)
    return selected


def main() -> None:
    if OUT_PATH.exists():
        LOGGER.info("Filtered descriptor file already exists: %s – skipping.", OUT_PATH)
        sys.exit(0)

    if not DESC_PATH.exists():
        LOGGER.error("Descriptor matrix not found. Run calc_descriptors.py first.")
        sys.exit(1)

    LOGGER.info("Reading descriptors from %s", DESC_PATH)
    df = pl.read_parquet(DESC_PATH)

    numeric_cols = [c for c in df.columns if c != "SMILES"]
    LOGGER.info("Initial number of descriptor columns: %d", len(numeric_cols))

    # Convert numeric part to pandas for efficient correlation computation
    df_num_pd = df.select(numeric_cols).to_pandas()

    selected_cols = greedy_correlation_filter(df_num_pd, threshold=CORR_THRESHOLD)
    LOGGER.info("Kept %d columns after correlation filter (thr=%.2f).", len(selected_cols), CORR_THRESHOLD)

    # Create final DataFrame and save
    df_selected = pl.concat([df.select("SMILES"), df.select(selected_cols)], how="horizontal")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_selected.write_parquet(OUT_PATH)
    LOGGER.info("Filtered descriptors saved to %s", OUT_PATH)


if __name__ == "__main__":
    main()
