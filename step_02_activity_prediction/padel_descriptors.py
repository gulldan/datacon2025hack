"""Optional PaDEL descriptor generation (T28) using **padelpy** wrapper.

Padelpy bundles PaDEL-Descriptor internally, so we avoid manual Java jar
invocation.  If ``config.USE_PADEL_DESCRIPTORS`` is True, this script loads the
processed activity dataset, computes descriptors for each SMILES via
``padelpy.from_smiles`` (multi-threaded), and writes them to
``results/padel_descriptors.parquet``.

The script is idempotent: it skips computation if the output file already
exists.

Reference: padelpy GitHub repo [ecrl/padelpy](https://github.com/ecrl/padelpy).
"""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd
import polars as pl
from padelpy import from_smiles  # type: ignore

import config
from utils.logger import LOGGER

OUT_PATH = config.PREDICTION_RESULTS_DIR / "padel_descriptors.parquet"


def batch_compute(smiles: list[str], threads: int = 4) -> pd.DataFrame:
    """Compute PaDEL descriptors for a list of SMILES using padelpy.

    padelpy handles batching internally; we simply forward the list.
    """
    LOGGER.info("Computing PaDEL descriptors for %d molecules (threads=%d)…", len(smiles), threads)
    desc_list: list[dict[str, Any]] = from_smiles(
        smiles,
        fingerprints=False,
        descriptors=True,
        threads=threads,
    )
    return pd.DataFrame(desc_list)


def main() -> None:
    if not getattr(config, "USE_PADEL_DESCRIPTORS", False):
        LOGGER.info("USE_PADEL_DESCRIPTORS is False – skipping PaDEL descriptor generation.")
        return

    if OUT_PATH.exists():
        LOGGER.info("PaDEL descriptor file already exists: %s – skipping.", OUT_PATH)
        return

    if not config.ACTIVITY_DATA_PROCESSED_PATH.exists():
        LOGGER.error("Processed dataset not found: %s", config.ACTIVITY_DATA_PROCESSED_PATH)
        sys.exit(1)

    df_proc = pl.read_parquet(config.ACTIVITY_DATA_PROCESSED_PATH)
    smiles_list = df_proc["SMILES"].to_list()

    # Compute descriptors
    try:
        df_padel_pd = batch_compute(smiles_list, threads=4)
    except Exception as exc:
        LOGGER.exception("PaDEL descriptor computation failed: %s", exc)
        sys.exit(1)

    # Ensure SMILES column present
    if "SMILES" not in df_padel_pd.columns:
        df_padel_pd.insert(0, "SMILES", smiles_list)  # type: ignore[arg-type]

    df_padel = pl.from_pandas(df_padel_pd)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_padel.write_parquet(OUT_PATH)
    LOGGER.info("PaDEL descriptors saved to %s", OUT_PATH)


if __name__ == "__main__":
    main()
