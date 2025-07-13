"""Calculate molecular descriptors for activity dataset (T27).

Outputs:
    • ``results/descriptors.parquet`` – table with SMILES and descriptor values

If the output file already exists it is reused (idempotent).

Notes:
-----
External libraries such as ``mordred`` and ``numpy`` sometimes emit benign
``SyntaxWarning`` and ``RuntimeWarning`` messages (for instance, overflow during
reduce operations on large arrays). These warnings do **not** affect the
descriptor values we actually use, but they can clutter the console output.

We globally suppress those specific categories to keep the pipeline logs
readable. Only critical errors will interrupt execution.
"""
from __future__ import annotations

import sys

# pylint: disable=wrong-import-position
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import polars as pl
from rdkit import Chem  # type: ignore
from rdkit.Chem import Descriptors  # type: ignore
from tqdm import tqdm  # type: ignore

import config
from utils.logger import LOGGER

# ---------------------------------------------------------------------------
# Silence non-critical warnings from third-party libraries
# ---------------------------------------------------------------------------

# 1) "is" with tuple literal inside Mordred – harmless, but noisy.
warnings.filterwarnings("ignore", category=SyntaxWarning)
# 2) Overflow or invalid operations inside NumPy reductions when Mordred
#    computes certain information-theoretic descriptors.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Also prevent NumPy from raising floating-point warnings that we already
# decided to silence above.
np.seterr(over="ignore", invalid="ignore")

# ---------------------------------------------------------------------------
# Prepare output path
# ---------------------------------------------------------------------------

OUT_PATH = config.PREDICTION_RESULTS_DIR / "descriptors.parquet"

# ---------------------------------------------------------------------------
# Load processed dataset
# ---------------------------------------------------------------------------

if not config.ACTIVITY_DATA_PROCESSED_PATH.exists():
    LOGGER.error("Processed dataset not found: %s", config.ACTIVITY_DATA_PROCESSED_PATH)
    sys.exit(1)

proc_df = pl.read_parquet(config.ACTIVITY_DATA_PROCESSED_PATH)
smiles_list: list[str] = proc_df["SMILES"].to_list()  # type: ignore[no-any-return]

LOGGER.info(f"Calculating RDKit descriptors for {len(smiles_list)} molecules…")

# RDKit 2D descriptors (~200, but many redundant); we'll take the official list (208)
RD_NAMES_FUNCS: list[tuple[str, Any]] = Descriptors.descList  # type: ignore[attr-defined]

# Build Mordred calculator (1826 descriptors) but filter 3D/slow ones via ignore_3D=True
try:
    from mordred import Calculator  # type: ignore
    from mordred import descriptors as mordred_desc

    calc = Calculator(mordred_desc, ignore_3D=True)
    use_mordred = True
except ImportError:  # fallback if mordred not installed
    LOGGER.warning("mordred package not installed – only RDKit descriptors will be computed.")
    use_mordred = False


# ---------------------------------------------------------------------------
# Incremental caching: skip molecules already processed
# ---------------------------------------------------------------------------

existing_df: pl.DataFrame | None = None
processed_smiles: set[str] = set()
if OUT_PATH.exists():
    existing_df = pl.read_parquet(OUT_PATH)
    processed_smiles = set(existing_df["SMILES"].to_list())  # type: ignore[no-any-return]
    LOGGER.info(f"Found existing descriptor file with {len(processed_smiles)} molecules; will compute {len(smiles_list) - len(processed_smiles)} new ones.")

smiles_to_compute = [s for s in smiles_list if s not in processed_smiles]


def compute_descriptors(smi: str) -> dict[str, Any] | None:
    """Compute RDKit (+ optional Mordred) descriptors for a single SMILES."""
    mol = Chem.MolFromSmiles(smi)  # type: ignore[attr-defined]
    if mol is None:
        return None

    rec: dict[str, Any] = {"SMILES": smi}

    # RDKit descriptors
    for name, func in RD_NAMES_FUNCS:
        try:
            val = func(mol)
            rec[name] = float(val) if val is not None else np.nan
        except Exception:
            rec[name] = np.nan

    if use_mordred:
        try:
            mord_vals = calc(mol)  # type: ignore[arg-type]
            for key, val in mord_vals.items():  # type: ignore[attr-defined]
                rec[str(key)] = float(val) if isinstance(val, (int, float)) else np.nan
        except Exception:
            # If Mordred fails for this molecule, leave NaNs; they'll be dropped later if all null
            pass

    return rec


LOGGER.info(f"Computing descriptors in parallel using {min(8, len(smiles_to_compute))} threads…")

records: list[dict[str, Any]] = []
if smiles_to_compute:
    with ThreadPoolExecutor(max_workers=min(8, len(smiles_to_compute))) as executor:
        for result in tqdm(executor.map(compute_descriptors, smiles_to_compute), total=len(smiles_to_compute)):
            if result is not None:
                records.append(result)

# Combine with any existing descriptors
if existing_df is not None and not existing_df.is_empty():
    df_existing = existing_df  # already a polars DataFrame
else:
    df_existing = None

if records:
    LOGGER.info("Building DataFrame for newly computed records…")
    df_new = pl.DataFrame(records).with_columns(
        [pl.col(col).cast(pl.Float64, strict=False) if col != "SMILES" else pl.col(col) for col in records[0].keys()]
    ).drop_nulls(subset=["SMILES"])

    df_desc = df_new if df_existing is None else pl.concat([df_existing, df_new], how="vertical")
else:
    # Nothing new computed
    if df_existing is None:
        LOGGER.error("No descriptors computed and no existing file present — aborting.")
        sys.exit(1)
    df_desc = df_existing

# After concatenation ensure columns with all NaN removed
numeric_cols = [c for c in df_desc.columns if c != "SMILES"]
non_null_cols = [c for c in numeric_cols if df_desc[c].null_count() < len(df_desc)]

df_desc = df_desc.select(["SMILES"] + non_null_cols)

# Zero variance filter
var_keep = []
for col in non_null_cols:
    if df_desc[col].n_unique() > 1:
        var_keep.append(col)

df_desc = df_desc.select(["SMILES"] + var_keep)

# Deduplicate in case SMILES duplicates existed
df_desc = df_desc.unique("SMILES")

LOGGER.info(f"Final descriptor matrix shape after merge/filter: {df_desc.shape}")

# Save to disk
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_desc.write_parquet(OUT_PATH)
LOGGER.info(f"Descriptors saved to {OUT_PATH}")
