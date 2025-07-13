"""Batch docking with AutoDock Vina.

Requires prepared receptor PDBQT and ligand PDBQT files (see `protein_prep.py`, `ligand_prep.py`).
Writes scores to `config.VINA_RESULTS_PATH`.
"""
from __future__ import annotations

import re
import shutil
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

# Early check for vina availability
VINA_AVAILABLE = shutil.which(VINA_BIN) is not None

# Detect whether this Vina build supports --log option
try:
    _help_out = subprocess.run([VINA_BIN, "--help"], check=True, capture_output=True, text=True)
    HAS_LOG_OPTION = "--log" in _help_out.stdout
except Exception:
    HAS_LOG_OPTION = False


def dock_ligand(lig_pdbqt: Path, out_pdbqt: Path, log_path: Path) -> float | None:
    global HAS_LOG_OPTION
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
    ]
    if HAS_LOG_OPTION:
        cmd += ["--log", str(log_path)]

    try:
        res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else ""
        # If the error is due to unknown --log option, switch flag off globally and retry once
        if "unrecognised option '--log'" in stderr or "unknown option log" in stderr:
            HAS_LOG_OPTION = False
            LOGGER.warning("Detected Vina without --log support. Re-running without log option.")
            return dock_ligand(lig_pdbqt, out_pdbqt, log_path)
        LOGGER.error(f"Vina execution failed for {lig_pdbqt.name}: {stderr.strip()} ({e})")
        return None
    except FileNotFoundError as e:
        LOGGER.error(f"Vina binary not found: {e}")
        return None

    # Parse score
    # 1) Try dedicated log file
    if HAS_LOG_OPTION and log_path.exists():
        for line in log_path.read_text().splitlines():
            if line.strip().startswith("1 "):
                parts = line.split()
                return float(parts[1])

    # 2) Try stdout of Vina
    out_text = res.stdout.decode()
    for line in out_text.splitlines():
        m = re.match(r"^\s*1\s+(-?\d+\.\d+)", line)
        if m:
            return float(m.group(1))

    # 3) Fallback: parse header of resulting PDBQT (REMARK VINA RESULT line)
    if out_pdbqt.exists():
        for line in out_pdbqt.read_text().splitlines():
            if line.startswith("REMARK VINA RESULT:"):
                # Example: "REMARK VINA RESULT:     -7.5      0.000      0.000"
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        return float(parts[3])
                    except ValueError:
                        continue
    return None


def has_atoms(pdbqt_path: Path) -> bool:
    try:
        with pdbqt_path.open() as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")):
                    return True
    except FileNotFoundError:
        return False
    return False


def main() -> None:
    receptor = config.PROTEIN_PDBQT_PATH
    if not receptor.exists():
        LOGGER.error(f"Receptor file {receptor} not found. Run protein_prep.py first.")
        _sys.exit(1)

    if not VINA_AVAILABLE:
        LOGGER.warning("AutoDock Vina binary not found – will skip docking and return no scores.")
        _sys.exit(1)

    lig_files = sorted(config.LIGAND_PDBQT_DIR.glob("*.pdbqt"))
    # keep only original ligand files (exclude previously docked outputs)
    lig_files = [lf for lf in lig_files if not lf.stem.endswith("_dock")]
    # keep only files that have atoms (skip empty/invalid files)
    lig_files = [lf for lf in lig_files if has_atoms(lf)]

    if not lig_files:
        LOGGER.error(f"No ligand PDBQT files found in {config.LIGAND_PDBQT_DIR}. Run ligand_prep.py first.")
        _sys.exit(1)

    results: list[tuple[str, float]] = []
    LOGGER.info(f"Docking {len(lig_files)} ligands with Vina…")
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
    LOGGER.info(f"Docking completed. Scores saved to {config.VINA_RESULTS_PATH}")


if __name__ == "__main__":
    main()
