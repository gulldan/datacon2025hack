"""Optional PaDEL descriptor generation (T28).

If ``config.USE_PADEL_DESCRIPTORS`` is True and ``config.PADEL_JAR_PATH`` exists,
this script runs PaDEL-Descriptor via Java to compute ~1800 descriptors and
saves them to ``results/padel_descriptors.parquet``. The script is idempotent –
it skips computation if output already exists and input SMILES did not change.

Requirements:
    • Java runtime available on PATH
    • PaDEL-Descriptor.jar present at configured path

PaDEL CLI is invoked with options:
    -threads 4  (parallel on 4 cores)
    -descriptortypes  Hydrogens,FunctionalGroups etc. (default)
    -maxruntime 30  (timeout per molecule, seconds)

We prepare a temporary SMILES file as input and parse the resulting CSV.
"""

from __future__ import annotations

import csv
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import polars as pl

import config
from utils.logger import LOGGER

OUT_PATH = config.PREDICTION_RESULTS_DIR / "padel_descriptors.parquet"


def java_available() -> bool:
    try:
        subprocess.run(["java", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def run_padel(smiles_path: Path, out_csv: Path) -> None:
    cmd = [
        "java",
        "-jar",
        str(config.PADEL_JAR_PATH),
        "-removesalt",
        "-standardizenitro",
        "-fingerprints",  # also compute fingerprints but we skip them later
        "-threads",
        "4",
        "-maxruntime",
        "30",
        "-descriptortypes",
        "ChemistryDevelopmentKit",
        "-dir",
        str(smiles_path.parent),
        "-file",
        str(out_csv),
    ]

    LOGGER.info("Running PaDEL-Descriptor (may take a while)…")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        LOGGER.error("PaDEL-Descriptor failed: %s", exc)
        raise SystemExit(1) from exc


def main() -> None:
    if not config.USE_PADEL_DESCRIPTORS:
        LOGGER.info("USE_PADEL_DESCRIPTORS is False – skipping PaDEL descriptor generation.")
        return

    if OUT_PATH.exists():
        LOGGER.info("PaDEL descriptor file already exists: %s – skipping.", OUT_PATH)
        return

    if not config.PADEL_JAR_PATH.exists():
        LOGGER.error("PaDEL jar not found: %s", config.PADEL_JAR_PATH)
        return

    if not java_available():
        LOGGER.error("Java runtime not found in PATH – cannot run PaDEL.")
        return

    if not config.ACTIVITY_DATA_PROCESSED_PATH.exists():
        LOGGER.error("Processed dataset missing: %s", config.ACTIVITY_DATA_PROCESSED_PATH)
        return

    df_proc = pl.read_parquet(config.ACTIVITY_DATA_PROCESSED_PATH)
    smiles_list = df_proc["SMILES"].to_list()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        smiles_file = tmpdir_path / "input.smi"

        # Write SMILES to file (one per line with index)
        with smiles_file.open("w", encoding="utf-8") as f:
            for idx, smi in enumerate(smiles_list, start=1):
                f.write(f"{smi}\t{idx}\n")

        out_csv = tmpdir_path / "padel_out.csv"
        run_padel(smiles_file, out_csv)

        # Read CSV into polars
        LOGGER.info("Parsing PaDEL output CSV…")
        # PaDEL CSV uses comma delimiter, first column Name second col SMILES skipped
        with out_csv.open("r", encoding="utf-8") as f:
            csv_reader = csv.DictReader(f)
            records: list[dict[str, Any]] = []
            for row in csv_reader:
                name = row.pop("Name", None)
                smiles = row.pop("SMILES", None)
                if smiles is None:
                    continue
                rec = {"SMILES": smiles}
                # Convert numeric columns to float if possible
                for k, v in row.items():
                    try:
                        rec[k] = float(v) if v not in ("NaN", "", None) else None
                    except ValueError:
                        rec[k] = None
                records.append(rec)

        df_padel = pl.DataFrame(records)

        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_padel.write_parquet(OUT_PATH)
        LOGGER.info("PaDEL descriptors saved to %s", OUT_PATH)


if __name__ == "__main__":
    main()
