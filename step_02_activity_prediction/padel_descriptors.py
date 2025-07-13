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

import math
import sys
from typing import Any

import pandas as pd
import polars as pl
from padelpy import from_smiles  # type: ignore

import config
from utils.logger import LOGGER

OUT_PATH = config.PREDICTION_RESULTS_DIR / "padel_descriptors.parquet"


def batch_compute(smiles: list[str], threads: int = 4, chunk_size: int = 500) -> pd.DataFrame:
    """Compute PaDEL descriptors in *chunks* to avoid global timeout.

    If a chunk fails due to timeout, it is recursively split into halves until
    the timeout is avoided or the chunk size becomes 1 (then the molecule is
    skipped with warning).
    """
    records: list[dict[str, Any]] = []

    total = len(smiles)
    n_chunks = math.ceil(total / chunk_size)

    for idx in range(n_chunks):
        sub = smiles[idx * chunk_size : (idx + 1) * chunk_size]

        csize = len(sub)
        attempt = 0
        while True:
            attempt += 1
            try:
                LOGGER.info(
                    f"PaDEL: chunk {idx + 1}/{n_chunks} (size={csize}, attempt={attempt}, threads={threads})…",
                    idx + 1,
                    n_chunks,
                    csize,
                    attempt,
                    threads,
                )

                desc_list: list[dict[str, Any]] = from_smiles(
                    sub,
                    fingerprints=False,
                    descriptors=True,
                    threads=threads,
                    timeout=600,  # 10-minute safety per chunk
                )
                records.extend(desc_list)
                break  # chunk succeeded
            except RuntimeError as exc:
                if "timed out" in str(exc) and csize > 1:
                    # Split chunk in half and retry
                    half = csize // 2
                    sub_first = sub[:half]
                    sub_second = sub[half:]
                    # Reinsert second half to process after first
                    smiles[idx * chunk_size : (idx + 1) * chunk_size] = sub_first
                    smiles.insert((idx + 1) * chunk_size, sub_second)  # type: ignore[arg-type]
                    csize = len(sub_first)
                    n_chunks += 1  # one extra chunk added
                    LOGGER.warning(f"PaDEL timeout – splitting chunk to {len(sub_first)} + {len(sub_second)}")
                    # loop continues with smaller chunk
                else:
                    LOGGER.error(f"PaDEL failed for chunk of size {csize}: {exc}")
                    if csize == 1:
                        break  # skip molecule
                    # else treat as skip for this chunk
                    break

    return pd.DataFrame(records)


def main() -> None:
    if not getattr(config, "USE_PADEL_DESCRIPTORS", False):
        LOGGER.info("USE_PADEL_DESCRIPTORS is False – skipping PaDEL descriptor generation.")
        return

    if OUT_PATH.exists():
        LOGGER.info(f"PaDEL descriptor file already exists: {OUT_PATH} – skipping.")
        return

    if not config.ACTIVITY_DATA_PROCESSED_PATH.exists():
        LOGGER.error(f"Processed dataset not found: {config.ACTIVITY_DATA_PROCESSED_PATH}")
        sys.exit(1)

    df_proc = pl.read_parquet(config.ACTIVITY_DATA_PROCESSED_PATH)
    smiles_list = df_proc["SMILES"].to_list()

    # Compute descriptors
    try:
        df_padel_pd = batch_compute(smiles_list, threads=4)
    except Exception as exc:
        LOGGER.exception(f"PaDEL descriptor computation failed: {exc}")
        sys.exit(1)

    # Ensure SMILES column present
    if "SMILES" not in df_padel_pd.columns:
        df_padel_pd.insert(0, "SMILES", smiles_list)  # type: ignore[arg-type]

    df_padel = pl.from_pandas(df_padel_pd)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_padel.write_parquet(OUT_PATH)
    LOGGER.info(f"PaDEL descriptors saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
