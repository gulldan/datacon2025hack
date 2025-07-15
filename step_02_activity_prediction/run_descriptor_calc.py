"""Wrapper script that leverages `descriptor_calculator.py` to compute and
cache RDKit, Mordred, PaDEL descriptors and several fingerprints in one go.

Output is a consolidated parquet file compatible with the rest of the
pipeline (same path/name as produced previously by `calc_descriptors.py`).

The heavy-lifting (parallelisation, caching, cleaning) is done inside
`descriptor_calculator` – this wrapper only orchestrates the calls and glues
all pieces together.
"""

from __future__ import annotations

import polars as pl

import config
from step_02_activity_prediction import descriptor_calculator as dc
from utils.logger import LOGGER


def main() -> None:
    # ------------------------------------------------------------------
    # 0. Load canonical SMILES from the processed activity dataset
    # ------------------------------------------------------------------
    proc_path = config.ACTIVITY_DATA_PROCESSED_PATH
    if not proc_path.exists():
        LOGGER.error(f"Processed dataset not found: {proc_path}")
        raise SystemExit(1)

    df_proc = pl.read_parquet(proc_path)
    smiles = df_proc["SMILES"].to_list()  # keep original order

    LOGGER.info(f"Descriptor pipeline started – {len(smiles)} molecules")

    # ------------------------------------------------------------------
    # 1. RDKit descriptors (always)
    # ------------------------------------------------------------------
    LOGGER.info(f"RDKit descriptors started – {len(smiles)} molecules")
    rd_df = dc.calculate_rdkit_descriptors(smiles, use_cache=True)

    # ------------------------------------------------------------------
    # 2. Mordred descriptors (optional, heavy)
    # ------------------------------------------------------------------
    LOGGER.info(f"Mordred descriptors started – {len(smiles)} molecules")
    mord_df = dc.calculate_mordred_descriptors(smiles, ignore_3d=True, use_cache=True)

    # ------------------------------------------------------------------
    # 3. PaDEL descriptors (optional – behind config flag & dependency)
    # ------------------------------------------------------------------
    padel_df = None
    if getattr(config, "USE_PADEL_DESCRIPTORS", False):
        LOGGER.info(f"PaDEL descriptors started – {len(smiles)} molecules")
        padel_df = dc.calculate_padel_descriptors(smiles, fingerprints=True, d_2d=True, d_3d=False, use_cache=True)

    # ------------------------------------------------------------------
    # 4. Fingerprints (Morgan, MACCS, etc.)
    # ------------------------------------------------------------------
    LOGGER.info(f"Fingerprints started – {len(smiles)} molecules")
    fp_dict = dc.calculate_fingerprints(smiles, use_cache=True)

    # ------------------------------------------------------------------
    # 5. Concatenate all pieces horizontally, preserving row order
    # ------------------------------------------------------------------
    LOGGER.info(f"Concatenating all pieces horizontally – {len(smiles)} molecules")

    def rename_if_collision(df: pl.DataFrame, existing_cols: set[str], prefix: str) -> pl.DataFrame:
        """Rename columns that already exist in *existing_cols* by adding *prefix*_"""
        rename_map: dict[str, str] = {}
        for col in df.columns:
            if col == "SMILES":
                continue
            if col in existing_cols:
                rename_map[col] = f"{prefix}_{col}"
        if rename_map:
            df = df.rename(rename_map)
        return df

    base_df = pl.DataFrame({"SMILES": smiles})
    pieces: list[pl.DataFrame] = [base_df]

    # Start with RDKit – baseline feature set
    pieces.append(rd_df)
    current_cols: set[str] = set(rd_df.columns)

    if mord_df is not None:
        mord_df = rename_if_collision(mord_df, current_cols, "mord")
        current_cols.update(mord_df.columns)
        pieces.append(mord_df)

    if padel_df is not None:
        padel_df = rename_if_collision(padel_df, current_cols, "padel")
        current_cols.update(padel_df.columns)
        pieces.append(padel_df)

    for name, fp_df in fp_dict.items():
        fp_df = rename_if_collision(fp_df, current_cols, name)
        current_cols.update(fp_df.columns)
        pieces.append(fp_df)

    full_df = pl.concat(pieces, how="horizontal")

    out_path = config.PREDICTION_RESULTS_DIR / "descriptors.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.write_parquet(out_path)

    LOGGER.info(f"Descriptors written to {out_path} (shape {full_df.shape})")


if __name__ == "__main__":
    main()
