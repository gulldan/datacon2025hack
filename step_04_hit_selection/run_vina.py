"""Batch docking with AutoDock Vina (CPU/GPU aware) **and automatic merge of docking
scores with the ligand‑ID↔SMILES map**.

* Reads prepared ligand PDBQT files (produced by `ligand_prep.py`).
* Docks every ligand against the target receptor (PDBQT) using AutoDock Vina (or
  AutoDock‑GPU / Vina‑GPU when available).
* Writes raw docking scores to ``config.VINA_RESULTS_PATH`` **and** a merged table
  (map + scores) to ``config.DOCKING_RESULTS_PATH`` (defaults to
  ``docking_results.parquet``).

The merge step loads the Parquet file created in the ligand‑prep stage (the file
must contain at least columns ``id_ligand`` and ``smiles``).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys as _sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import polars as pl
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Project imports & basic setup
# -----------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in _sys.path:
    _sys.path.insert(0, str(ROOT_DIR))

import config  # type: ignore
from utils.logger import LOGGER as logger  # type: ignore

# -----------------------------------------------------------------------------
# Constants & binary availability checks
# -----------------------------------------------------------------------------

VINA_BIN = "vina"  # assume in PATH or specified via environment / aliases
VINA_AVAILABLE = shutil.which(VINA_BIN) is not None

AUTODOCK_GPU_AVAILABLE = Path(config.DOCKING_PARAMETERS.get("autodock_gpu_path", "")).exists()
VINA_GPU_AVAILABLE = Path(config.DOCKING_PARAMETERS.get("vina_gpu_path", "")).exists()
GPU_AVAILABLE = AUTODOCK_GPU_AVAILABLE or VINA_GPU_AVAILABLE

# Detect `--log` support once (Vina 1.2+)
try:
    _help_out = subprocess.run([VINA_BIN, "--help"], check=True, capture_output=True, text=True)
    HAS_LOG_OPTION = "--log" in _help_out.stdout
except Exception:  # pylint: disable=broad-except
    HAS_LOG_OPTION = False

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def dock_ligand_cpu(lig_pdbqt: Path, out_pdbqt: Path, log_path: Path, docking_params: dict) -> float | None:
    """Run AutoDock Vina (CPU) for a single ligand and parse the best score."""
    global HAS_LOG_OPTION  # noqa: PLW0603

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
        "--exhaustiveness",
        str(docking_params["exhaustiveness"]),
        "--num_modes",
        str(docking_params["num_modes"]),
        "--energy_range",
        str(docking_params["energy_range"]),
        "--out",
        str(out_pdbqt),
        "--cpu",
        str(docking_params["num_threads"]),
    ]
    if HAS_LOG_OPTION:
        cmd += ["--log", str(log_path)]

    try:
        res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode() if exc.stderr else ""
        if "unrecognised option '--log'" in stderr or "unknown option log" in stderr:
            HAS_LOG_OPTION = False
            logger.warning("Vina build without --log detected — retrying without log option …")
            return dock_ligand_cpu(lig_pdbqt, out_pdbqt, log_path, docking_params)
        logger.error(f"Vina failed for {lig_pdbqt.name}: {stderr.strip()}")
        return None
    except FileNotFoundError as exc:
        logger.error(f"Vina binary not found: {exc}")
        return None

    # ---------------------------------------------------------------------
    # Parsing score hierarchy: log → stdout → PDBQT REMARK
    # ---------------------------------------------------------------------
    if HAS_LOG_OPTION and log_path.exists():
        score = _extract_score_from_log(log_path.read_text())
        if score is not None:
            return score

    score = _extract_score_from_stdout(res.stdout.decode())
    if score is not None:
        return score

    if out_pdbqt.exists():
        score = _extract_score_from_pdbqt(out_pdbqt.read_text())
        if score is not None:
            return score

    logger.warning(f"No score found for {lig_pdbqt.name}")
    return None


def _extract_score_from_log(text: str) -> float | None:
    for line in text.splitlines():
        if line.strip().startswith("1 "):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    pass
    return None


def _extract_score_from_stdout(text: str) -> float | None:
    for line in text.splitlines():
        m = re.match(r"^\s*1\s+(-?\d+\.\d+)", line)
        if m:
            return float(m.group(1))
    return None


def _extract_score_from_pdbqt(text: str) -> float | None:
    for line in text.splitlines():
        if line.startswith("REMARK VINA RESULT:"):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    return float(parts[3])
                except ValueError:
                    pass
    return None


def optimize_vina_parameters() -> dict:
    """Return a parameter dict tuned for the machine."""
    cores = cpu_count()
    params = config.DOCKING_PARAMETERS.copy()

    if cores >= 16:
        params["num_threads"] = min(cores // 4, 8)
    elif cores >= 8:
        params["num_threads"] = cores // 2
    else:
        params["num_threads"] = max(1, cores // 2)

    logger.info(
        f"Vina params: threads={params['num_threads']}, exhaustiveness={params['exhaustiveness']}, modes={params['num_modes']}"
    )
    return params


def _dock_worker(args: tuple[Path, Path, Path, dict]) -> tuple[str, float | None]:
    lig, out_pdbqt, log_path, params = args
    score = dock_ligand_cpu(lig, out_pdbqt, log_path, params)
    return lig.stem, score


def dock_ligands_parallel(lig_files: list[Path], max_workers: int | None = None) -> list[tuple[str, float]]:
    if max_workers is None:
        max_workers = min(cpu_count(), len(lig_files))

    logger.info(f"Parallel docking with {max_workers} processes")

    params = optimize_vina_parameters()

    tasks = []
    for lig in lig_files:
        tasks.append(
            (
                lig,
                lig.with_name(lig.stem + "_dock.pdbqt"),
                lig.with_suffix(".log"),
                params,
            )
        )

    results: list[tuple[str, float | None]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(_dock_worker, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(future_map), total=len(tasks), desc="Docking"):
            lig_path = future_map[fut]
            try:
                lig_id, score = fut.result()
                if score is not None:
                    results.append((lig_id, score))
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(f"Docking failed for {lig_path}: {exc}")
    return results


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def has_atoms(pdbqt_path: Path) -> bool:
    try:
        with pdbqt_path.open(encoding="utf-8") as fh:
            return any(line.startswith(("ATOM", "HETATM")) for line in fh)
    except FileNotFoundError:
        return False


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def main() -> None:
    if not getattr(config, "USE_VINA_DOCKING", True):
        logger.info("USE_VINA_DOCKING is False — skipping docking stage")
        return

    if not VINA_AVAILABLE:
        logger.error("AutoDock Vina binary not found in PATH — aborting")
        _sys.exit(1)

    receptor = config.PROTEIN_PDBQT_PATH
    if not receptor.exists():
        logger.error(f"Receptor PDBQT ({receptor}) not found. Run protein_prep.py first.")
        _sys.exit(1)

    lig_files = sorted(config.LIGAND_PDBQT_DIR.glob("*.pdbqt"))
    lig_files = [p for p in lig_files if not p.stem.endswith("_dock")]
    lig_files = [p for p in lig_files if has_atoms(p)]

    if not lig_files:
        logger.error(f"No valid ligand PDBQT files in {config.LIGAND_PDBQT_DIR}")
        _sys.exit(1)

    logger.info(f"Docking {len(lig_files)} ligands …")
    t0 = time.time()
    results = dock_ligands_parallel(lig_files)
    elapsed = time.time() - t0
    logger.info(f"Docking finished in {elapsed:.2f} s")

    if not results:
        logger.warning("No scores collected — nothing to merge")
        return

    # ---------------------------------------------------------------------
    # Save raw scores
    # ---------------------------------------------------------------------
    df_scores = pl.DataFrame(results, schema=["id_ligand", "docking_score"])
    df_scores.write_parquet(config.VINA_RESULTS_PATH)
    logger.info(f"Raw scores saved → {config.VINA_RESULTS_PATH}")

    # ---------------------------------------------------------------------
    # Merge with ligand‑ID map for downstream analysis
    # ---------------------------------------------------------------------
    mapping_path = config.LIGAND_MAPPING_PATH
    if not mapping_path.exists():
        logger.warning(f"Mapping file {mapping_path} not found — skipping merge step")
        return

    df_map = pl.read_parquet(mapping_path)
    if "id_ligand" not in df_map.columns:
        logger.error(f"Mapping file {mapping_path} lacks 'id_ligand' column — cannot merge")
        return

    df_merged = df_map.join(df_scores, on="id_ligand", how="left")

    docking_results_path = config.DOCKING_RESULTS_PATH
    df_merged.write_parquet(docking_results_path)

    logger.info(f"Merged docking results saved → {docking_results_path} ({len(df_merged)} entries, scored: {df_scores.height})")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
