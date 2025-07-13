"""Inference script for DYRK1A activity predictor.

Usage examples:

1. Predict from a CSV file with a column `smiles` and write out new CSV:

    uv run python step_02_activity_prediction/predict_activity.py --input molecules.csv --output predictions.csv

2. Predict a list of SMILES passed via CLI:

    uv run python step_02_activity_prediction/predict_activity.py "CCO" "c1ccccc1O"

The script uses *coefficients* stored in ``config.MODEL_PATH`` (npz) and the helper
functions in ``model_utils`` to compute Morgan fingerprints and linear predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
from loguru import logger

from step_02_activity_prediction.model_utils import load_model, predict_smiles

# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DYRK1A activity predictor (pIC50)")
    parser.add_argument("smiles", nargs="*", help="SMILES strings to predict (mutually exclusive with --input)")
    parser.add_argument("--input", "-i", type=Path, help="Path to CSV/Parquet file with column 'smiles'")
    parser.add_argument("--output", "-o", type=Path, help="Destination file to save predictions (CSV/Parquet by extension)")
    return parser.parse_args()


def read_smiles_from_file(path: Path) -> list[str]:
    """Read *smiles* column from CSV/Parquet (auto-detected by extension)."""
    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        df = pl.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pl.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")
    if "smiles" not in df.columns:
        raise ValueError("Input file must contain column 'smiles'")
    return df["smiles"].to_list()  # type: ignore[no-any-return]


def write_predictions(df: pl.DataFrame, out_path: Path) -> None:
    """Write dataframe *df* to *out_path* (format by extension)."""
    if out_path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        df.write_csv(out_path)
    elif out_path.suffix.lower() in {".parquet", ".pq"}:
        df.write_parquet(out_path)
    else:
        raise ValueError(f"Unsupported output extension: {out_path.suffix}")


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.input is None and not args.smiles:
        logger.error("Provide SMILES via positional arguments or --input file path.")
        sys.exit(1)

    # 1. Collect SMILES list
    if args.input:
        smiles_list = read_smiles_from_file(args.input)
        logger.info(f"Loaded {len(smiles_list)} SMILES from {args.input}")
    else:
        smiles_list = args.smiles
        logger.info(f"Received {len(smiles_list)} SMILES via CLI")

    # 2. Predict
    model = load_model()
    preds = predict_smiles(smiles_list, model)

    # 3. Prepare dataframe
    result_df = pl.DataFrame({"smiles": smiles_list, "predicted_pIC50": preds})
    logger.info(f"Example predictions:\n{result_df.head()}")

    # 4. Output
    if args.output:
        write_predictions(result_df, args.output)
        logger.info(f"Predictions written to {args.output}")
    else:
        # Print to stdout as CSV
        sys.stdout.write(result_df.write_csv())


if __name__ == "__main__":
    main()
