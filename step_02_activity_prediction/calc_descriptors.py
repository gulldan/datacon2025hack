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
from typing import Any

import numpy as np
import polars as pl
from rdkit import Chem  # type: ignore
from rdkit.Chem import Descriptors  # type: ignore

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

if OUT_PATH.exists():
    LOGGER.info("Descriptor file already exists: %s — skipping.", OUT_PATH)
    sys.exit(0)

# ---------------------------------------------------------------------------
# Load processed dataset
# ---------------------------------------------------------------------------

if not config.ACTIVITY_DATA_PROCESSED_PATH.exists():
    LOGGER.error("Processed dataset not found: %s", config.ACTIVITY_DATA_PROCESSED_PATH)
    sys.exit(1)

proc_df = pl.read_parquet(config.ACTIVITY_DATA_PROCESSED_PATH)
smiles_list: list[str] = proc_df["SMILES"].to_list()  # type: ignore[no-any-return]

LOGGER.info("Calculating RDKit descriptors for %d molecules…", len(smiles_list))

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

records: list[dict[str, Any]] = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)  # type: ignore[attr-defined]
    if mol is None:
        continue
    rec: dict[str, Any] = {"SMILES": smi}

    # RDKit
    for name, func in RD_NAMES_FUNCS:
        try:
            val = func(mol)
            # ensure numeric
            rec[name] = float(val) if val is not None else np.nan
        except Exception:
            rec[name] = np.nan

    # Mordred
    if use_mordred:
        try:
            mord_vals = calc(mol)  # type: ignore[arg-type]
            for key, val in mord_vals.items():  # type: ignore[attr-defined]
                # Some values are strings / None
                rec[str(key)] = float(val) if isinstance(val, (int, float)) else np.nan
        except Exception:
            # If Mordred fails for molecule, fill NaNs later
            pass

    records.append(rec)

# Convert to polars and drop columns with all NaN
LOGGER.info("Building DataFrame…")
df_desc = pl.DataFrame(records).with_columns(
    [pl.col(col).cast(pl.Float64, strict=False) if col != "SMILES" else pl.col(col) for col in records[0].keys()]
).drop_nulls(subset=["SMILES"])

# Remove columns that are all null or have zero variance
numeric_cols = [c for c in df_desc.columns if c != "SMILES"]
non_null_cols = [c for c in numeric_cols if df_desc[c].null_count() < len(df_desc)]

df_desc = df_desc.select(["SMILES"] + non_null_cols)

# Zero variance filter
var_keep = []
for col in non_null_cols:
    if df_desc[col].n_unique() > 1:
        var_keep.append(col)

df_desc = df_desc.select(["SMILES"] + var_keep)

LOGGER.info("Final descriptor matrix shape: %s", df_desc.shape)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_desc.write_parquet(OUT_PATH)
LOGGER.info("Descriptors saved to %s", OUT_PATH)
