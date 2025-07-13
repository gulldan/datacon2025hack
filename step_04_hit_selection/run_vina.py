"""Batch docking with AutoDock Vina.

Requires prepared receptor PDBQT and ligand PDBQT files (see `protein_prep.py`, `ligand_prep.py`).
Writes scores to `config.VINA_RESULTS_PATH`.
"""
from __future__ import annotations

import subprocess
import sys as _sys
from pathlib import Path

import polars as pl

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in _sys.path:
    _sys.path.insert(0, str(ROOT_DIR))

import config
from utils.logger import LOGGER

VINA_BIN = "vina"  # assume in PATH


def dock_ligand(lig_pdbqt: Path, out_pdbqt: Path, log_path: Path) -> float | None:
    cmd = [
        VINA_BIN,
        "--receptor",
        str(config.PROTEIN_PDBQT_PATH),
        "--ligand",
        str(lig_pdbqt),
        "--center_x",
        str(config.BOX_CENTER[0]),
        "--center_y",
        str(config.BOX_CENTER[1]),
        "--center_z",
        str(config.BOX_CENTER[2]),
        "--size_x",
        str(config.BOX_SIZE[0]),
        "--size_y",
        str(config.BOX_SIZE[1]),
        "--size_z",
        str(config.BOX_SIZE[2]),
        "--out",
        str(out_pdbqt),
        "--log",
        str(log_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        LOGGER.error("Vina failed for %s: %s", lig_pdbqt.name, e)
        return None

    # Parse score from log
    try:
        for line in log_path.read_text().splitlines():
            if line.strip().startswith("1 "):  # first pose
                parts = line.split()
                return float(parts[1])
    except Exception:
        pass
    return None


def main() -> None:
    receptor = config.PROTEIN_PDBQT_PATH
    if not receptor.exists():
        LOGGER.error("Receptor file %s not found. Run protein_prep.py first.", receptor)
        return

    lig_files = sorted(config.LIGAND_PDBQT_DIR.glob("*.pdbqt"))
    if not lig_files:
        LOGGER.error("No ligand PDBQT files found in %s. Run ligand_prep.py first.", config.LIGAND_PDBQT_DIR)
        return

    results: list[tuple[str, float]] = []
    LOGGER.info("Docking %d ligands with Vina…", len(lig_files))
    for lig in lig_files:
        out_pdbqt = lig.with_name(lig.stem + "_dock.pdbqt")
        log_path = lig.with_suffix(".log")
        score = dock_ligand(lig, out_pdbqt, log_path)
        if score is not None:
            results.append((lig.stem, score))

    if not results:
        LOGGER.warning("No docking scores obtained.")
        return

    df = pl.DataFrame(results, schema=["ligand_id", "docking_score"])
    df.write_parquet(config.VINA_RESULTS_PATH)
    LOGGER.info("Docking completed. Scores saved to %s", config.VINA_RESULTS_PATH)


if __name__ == "__main__":
    main()
