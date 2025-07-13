"""Batch docking with AutoDock Vina and GPU acceleration.

Requires prepared receptor PDBQT and ligand PDBQT files (see `protein_prep.py`, `ligand_prep.py`).
Writes scores to `config.VINA_RESULTS_PATH`.
Supports GPU acceleration via AutoDock-GPU and Vina-GPU.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys as _sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import polars as pl
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in _sys.path:
    _sys.path.insert(0, str(ROOT_DIR))

import config
from utils.logger import LOGGER as logger

VINA_BIN = "vina"  # assume in PATH

# Early check for vina availability
VINA_AVAILABLE = shutil.which(VINA_BIN) is not None

# GPU docking availability checks
AUTODOCK_GPU_AVAILABLE = Path(config.DOCKING_PARAMETERS.get("autodock_gpu_path", "")).exists()
VINA_GPU_AVAILABLE = Path(config.DOCKING_PARAMETERS.get("vina_gpu_path", "")).exists()
GPU_AVAILABLE = AUTODOCK_GPU_AVAILABLE or VINA_GPU_AVAILABLE

# Detect whether this Vina build supports --log option
try:
    _help_out = subprocess.run([VINA_BIN, "--help"], check=True, capture_output=True, text=True)
    HAS_LOG_OPTION = "--log" in _help_out.stdout
except Exception:
    HAS_LOG_OPTION = False


def dock_ligand_gpu(lig_pdbqt: Path, out_pdbqt: Path, log_path: Path, docking_params: dict | None = None) -> float | None:
    """GPU-accelerated docking using AutoDock-GPU, Vina-GPU, or optimized CPU"""
    if docking_params is None:
        docking_params = optimize_vina_parameters()

    gpu_engine = docking_params.get("gpu_engine", "vina_optimized")

    if gpu_engine == "autodock_gpu" and AUTODOCK_GPU_AVAILABLE:
        return dock_with_autodock_gpu(lig_pdbqt, out_pdbqt, log_path, docking_params)
    if gpu_engine == "vina_gpu" and VINA_GPU_AVAILABLE:
        return dock_with_vina_gpu(lig_pdbqt, out_pdbqt, log_path, docking_params)
    if gpu_engine == "vina_optimized":
        return dock_with_vina_optimized(lig_pdbqt, out_pdbqt, log_path, docking_params)
    logger.warning(f"GPU engine {gpu_engine} not available, falling back to CPU Vina")
    return dock_ligand_cpu(lig_pdbqt, out_pdbqt, log_path, docking_params)


def dock_with_autodock_gpu(lig_pdbqt: Path, out_pdbqt: Path, log_path: Path, docking_params: dict) -> float | None:
    """Docking with AutoDock-GPU"""
    autodock_gpu_bin = config.DOCKING_PARAMETERS["autodock_gpu_path"]

    # Create temporary directory for AutoDock-GPU files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Prepare grid maps for AutoDock-GPU
        # NOTE: AutoDock-GPU requires grid maps (.fld file) which need to be prepared beforehand
        # For now, we'll use a simplified approach by creating a simple grid file
        fld_path = temp_path / "receptor.maps.fld"
        create_fld_file(fld_path, docking_params)

        # Run AutoDock-GPU with modern parameters
        cmd = [
            str(autodock_gpu_bin),
            "--lfile", str(lig_pdbqt),
            "--ffile", str(fld_path),
            "--resnam", lig_pdbqt.stem,
            "--devnum", str(docking_params.get("gpu_device", 0) + 1),  # AutoDock-GPU uses 1-based indexing
            "--nrun", str(docking_params.get("autodock_gpu_nrun", 20)),
            "--nev", str(docking_params.get("autodock_gpu_nev", 2500000)),
            "--ngen", str(docking_params.get("autodock_gpu_ngen", 42000)),
            "--psize", str(docking_params.get("autodock_gpu_psize", 150)),
            "--heuristics", str(docking_params.get("autodock_gpu_heuristics", 1)),
            "--autostop", str(docking_params.get("autodock_gpu_autostop", 1)),
            "--xmloutput", str(docking_params.get("autodock_gpu_xml_output", 1)),
            "--dlgoutput", str(docking_params.get("autodock_gpu_dlg_output", 1))
        ]

        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True,
                                    timeout=docking_params.get("timeout_per_ligand", 60),
                                    cwd=temp_path)

            if result.returncode == 0:
                # Parse the output for binding energy from DLG file
                dlg_path = temp_path / f"{lig_pdbqt.stem}.dlg"
                return parse_autodock_gpu_output(result.stdout, log_path, dlg_path)
            logger.error(f"AutoDock-GPU failed for {lig_pdbqt.name}: {result.stderr}")
            return None

        except subprocess.TimeoutExpired:
            logger.warning(f"AutoDock-GPU timeout for {lig_pdbqt.name}")
            return None
        except Exception as e:
            logger.error(f"AutoDock-GPU error for {lig_pdbqt.name}: {e}")
            return None


def dock_with_vina_optimized(lig_pdbqt: Path, out_pdbqt: Path, log_path: Path, docking_params: dict) -> float | None:
    """Optimized CPU Vina docking with GPU-like performance settings"""
    # Агрессивные настройки для максимальной производительности
    optimized_params = docking_params.copy()

    # Увеличиваем parallelism для имитации GPU
    optimized_params["num_threads"] = min(cpu_count(), 16)  # Максимальное использование CPU
    optimized_params["exhaustiveness"] = 64  # Высокая точность поиска
    optimized_params["num_modes"] = 50  # Больше режимов для лучшего поиска
    optimized_params["energy_range"] = 5.0  # Широкий диапазон энергий

    logger.debug(f"Optimized GPU-like docking for {lig_pdbqt.name}: threads={optimized_params['num_threads']}, exhaustiveness={optimized_params['exhaustiveness']}")

    # Используем стандартную CPU функцию с оптимизированными параметрами
    return dock_ligand_cpu(lig_pdbqt, out_pdbqt, log_path, optimized_params)


def dock_with_vina_gpu(lig_pdbqt: Path, out_pdbqt: Path, log_path: Path, docking_params: dict) -> float | None:
    """Docking with Vina-GPU"""
    vina_gpu_bin = config.DOCKING_PARAMETERS["vina_gpu_path"]

    cmd = [
        str(vina_gpu_bin),
        "--receptor", str(config.PROTEIN_PDBQT_PATH),
        "--ligand", str(lig_pdbqt),
        "--center_x", str(config.BOX_CENTER[0]),
        "--center_y", str(config.BOX_CENTER[1]),
        "--center_z", str(config.BOX_CENTER[2]),
        "--size_x", str(config.BOX_SIZE[0]),
        "--size_y", str(config.BOX_SIZE[1]),
        "--size_z", str(config.BOX_SIZE[2]),
        "--exhaustiveness", str(docking_params["exhaustiveness"]),
        "--num_modes", str(docking_params["num_modes"]),
        "--energy_range", str(docking_params["energy_range"]),
        "--out", str(out_pdbqt),
        "--opencl_binary_path", str(Path(vina_gpu_bin).parent),
        "--thread", str(docking_params["num_threads"]),
        "--opencl_device_id", str(docking_params.get("gpu_device", 0))
    ]

    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=docking_params.get("timeout_per_ligand", 60))

        if result.returncode == 0:
            # Parse the output for binding energy
            return parse_vina_gpu_output(result.stdout, log_path)
        logger.error(f"Vina-GPU failed for {lig_pdbqt.name}: {result.stderr}")
        return None

    except subprocess.TimeoutExpired:
        logger.warning(f"Vina-GPU timeout for {lig_pdbqt.name}")
        return None
    except Exception as e:
        logger.error(f"Vina-GPU error for {lig_pdbqt.name}: {e}")
        return None


def create_fld_file(fld_path: Path, docking_params: dict) -> None:
    """Create Field File for AutoDock-GPU"""
    # Create a basic .fld file structure
    fld_content = f"""#SPACING 0.375
#NELEMENTS 6
#CENTER {config.BOX_CENTER[0]} {config.BOX_CENTER[1]} {config.BOX_CENTER[2]}
#MACROMOLECULE {config.PROTEIN_PDBQT_PATH}
#GRID_PARAMETER_FILE receptor.gpf
#
"""
    fld_path.write_text(fld_content)


def create_dpf_file(dpf_path: Path, lig_pdbqt: Path, out_pdbqt: Path, docking_params: dict) -> None:
    """Create Docking Parameter File for AutoDock-GPU"""
    dpf_content = f"""autodock_parameter_version 4.2
outlev 1
intelec
seed pid time
ligand_types A C HD N OA SA
fld receptor.maps.fld
map receptor.A.map
map receptor.C.map
map receptor.HD.map
map receptor.N.map
map receptor.OA.map
map receptor.SA.map
elecmap receptor.e.map
dsolvmap receptor.d.map
move {lig_pdbqt}
about {config.BOX_CENTER[0]} {config.BOX_CENTER[1]} {config.BOX_CENTER[2]}
tran0 random
axisangle0 random
dihe0 random
tstep 2.0
qstep 50.0
dstep 50.0
torsdof 14
rmstol 2.0
extnrg 1000.0
e0max 0.0 10000
ga_pop_size 150
ga_num_evals {docking_params.get('exhaustiveness', 8) * 2500}
ga_num_generations 27000
ga_elitism 1
ga_mutation_rate 0.02
ga_crossover_rate 0.8
ga_window_size 10
ga_cauchy_alpha 0.0
ga_cauchy_beta 1.0
set_ga
sw_max_its 300
sw_max_succ 4
sw_max_fail 4
sw_rho 1.0
sw_lb_rho 0.01
ls_search_freq 0.06
set_psw1
unbound_model bound
ga_run {docking_params.get('num_modes', 20)}
analysis
"""
    dpf_path.write_text(dpf_content)


def parse_autodock_gpu_output(output: str, log_path: Path, dlg_path: Path | None = None) -> float | None:
    """Parse AutoDock-GPU output for binding energy"""
    # Save full output to log
    log_path.write_text(output)

    # Try parsing from DLG file first if it exists
    if dlg_path and dlg_path.exists():
        try:
            dlg_content = dlg_path.read_text()
            lines = dlg_content.split("\n")
            for line in lines:
                if "DOCKED:" in line and "ENERGY" in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            return float(parts[3])
                        except ValueError:
                            continue
        except Exception as e:
            logger.warning(f"Could not parse DLG file {dlg_path}: {e}")

    # Look for binding energy in stdout output
    lines = output.split("\n")
    for line in lines:
        if "LOWEST BINDING ENERGY" in line or "Final Docked Energy" in line:
            try:
                # Extract energy value
                energy_match = re.search(r"[-+]?\d*\.?\d+", line)
                if energy_match:
                    return float(energy_match.group())
            except ValueError:
                continue

    # Alternative parsing for different output formats
    for line in lines:
        if line.strip().startswith("1"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    continue

    logger.warning(f"Could not parse AutoDock-GPU output: {output[:200]}...")
    return None


def parse_vina_gpu_output(output: str, log_path: Path) -> float | None:
    """Parse Vina-GPU output for binding energy"""
    # Save full output to log
    log_path.write_text(output)

    # Look for binding energy in output (similar to regular Vina)
    lines = output.split("\n")
    for line in lines:
        if line.strip().startswith("1"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    continue

    logger.warning(f"Could not parse Vina-GPU output: {output[:200]}...")
    return None


def dock_ligand(lig_pdbqt: Path, out_pdbqt: Path, log_path: Path, docking_params: dict | None = None) -> float | None:
    """Main docking function that chooses between GPU and CPU based on config"""
    if docking_params is None:
        docking_params = optimize_vina_parameters()

    # Check if GPU docking is enabled and available
    if docking_params.get("use_gpu", False) and GPU_AVAILABLE:
        return dock_ligand_gpu(lig_pdbqt, out_pdbqt, log_path, docking_params)
    return dock_ligand_cpu(lig_pdbqt, out_pdbqt, log_path, docking_params)


def dock_ligand_cpu(lig_pdbqt: Path, out_pdbqt: Path, log_path: Path, docking_params: dict | None = None) -> float | None:
    global HAS_LOG_OPTION
    # Используем переданные параметры или получаем оптимизированные один раз
    if docking_params is None:
        docking_params = optimize_vina_parameters()

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
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else ""
        # If the error is due to unknown --log option, switch flag off globally and retry once
        if "unrecognised option '--log'" in stderr or "unknown option log" in stderr:
            HAS_LOG_OPTION = False
            logger.warning("Detected Vina without --log support. Re-running without log option.")
            return dock_ligand_cpu(lig_pdbqt, out_pdbqt, log_path, docking_params)
        logger.error(f"Vina execution failed for {lig_pdbqt.name}: {stderr.strip()} ({e})")
        return None
    except FileNotFoundError as e:
        logger.error(f"Vina binary not found: {e}")
        return None

    # Parse score
    # 1) Try dedicated log file
    if HAS_LOG_OPTION and log_path.exists():
        log_content = log_path.read_text()
        logger.debug(f"Парсинг лог файла {log_path}: {log_content[:200]}...")
        for line in log_content.splitlines():
            if line.strip().startswith("1 "):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        score = float(parts[1])
                        logger.debug(f"Найден скор в логе: {score}")
                        return score
                    except ValueError:
                        continue

    # 2) Try stdout of Vina
    out_text = res.stdout.decode()
    logger.debug(f"Парсинг stdout Vina: {out_text[:200]}...")
    for line in out_text.splitlines():
        m = re.match(r"^\s*1\s+(-?\d+\.\d+)", line)
        if m:
            score = float(m.group(1))
            logger.debug(f"Найден скор в stdout: {score}")
            return score

    # 3) Fallback: parse header of resulting PDBQT (REMARK VINA RESULT line)
    if out_pdbqt.exists():
        pdbqt_content = out_pdbqt.read_text()
        logger.debug(f"Парсинг PDBQT файла {out_pdbqt}: {pdbqt_content[:200]}...")
        for line in pdbqt_content.splitlines():
            if line.startswith("REMARK VINA RESULT:"):
                # Example: "REMARK VINA RESULT:     -7.5      0.000      0.000"
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        score = float(parts[3])
                        logger.debug(f"Найден скор в PDBQT: {score}")
                        return score
                    except ValueError:
                        continue

    logger.warning(f"Не удалось найти скор для {lig_pdbqt.name}")
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


def dock_ligand_wrapper(args: tuple[Path, Path, Path, dict]) -> tuple[str, float | None]:
    """Wrapper function for parallel docking"""
    lig_pdbqt, out_pdbqt, log_path, docking_params = args
    score = dock_ligand(lig_pdbqt, out_pdbqt, log_path, docking_params)
    return (lig_pdbqt.stem, score)


def dock_ligands_parallel(lig_files: list[Path], max_workers: int | None = None) -> list[tuple[str, float]]:
    """Parallel docking of multiple ligands"""
    if max_workers is None:
        # Используем количество CPU ядер, но не более чем количество лигандов
        max_workers = min(cpu_count(), len(lig_files))

    logger.info(f"Запуск параллельного докинга с {max_workers} процессами")

    # Оптимизируем параметры один раз для всех лигандов
    docking_params = optimize_vina_parameters()

    # Подготавливаем аргументы для параллельной обработки
    dock_args = []
    for lig in lig_files:
        out_pdbqt = lig.with_name(lig.stem + "_dock.pdbqt")
        log_path = lig.with_suffix(".log")
        dock_args.append((lig, out_pdbqt, log_path, docking_params))

    results = []

    # Используем ProcessPoolExecutor для CPU-интенсивных задач
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Отправляем задачи в пул
        future_to_ligand = {executor.submit(dock_ligand_wrapper, args): args[0] for args in dock_args}

        # Собираем результаты с прогресс-баром
        for future in tqdm(as_completed(future_to_ligand), total=len(dock_args), desc="Докинг"):
            ligand_file = future_to_ligand[future]
            try:
                ligand_name, score = future.result()
                if score is not None:
                    results.append((ligand_name, score))
            except Exception as e:
                logger.error(f"Ошибка при докинге {ligand_file}: {e}")

    return results


def dock_ligands_batch(lig_files: list[Path], batch_size: int = 1000) -> list[tuple[str, float]]:
    """Batch docking for large datasets"""
    all_results = []

    for i in range(0, len(lig_files), batch_size):
        batch = lig_files[i:i + batch_size]
        logger.info(f"Обработка батча {i//batch_size + 1}: {len(batch)} лигандов")

        batch_results = dock_ligands_parallel(batch)
        all_results.extend(batch_results)

        # Сохраняем промежуточные результаты
        if batch_results:
            df_batch = pl.DataFrame(batch_results, schema=["ligand_id", "docking_score"])
            batch_path = config.VINA_RESULTS_PATH.with_name(f"vina_batch_{i//batch_size + 1}.parquet")
            df_batch.write_parquet(batch_path)
            logger.info(f"Промежуточные результаты сохранены: {batch_path}")

    return all_results


def optimize_vina_parameters() -> dict:
    """Оптимизация параметров Vina на основе доступных ресурсов и GPU настроек"""
    cpu_cores = cpu_count()

    # Базовые параметры из конфига
    params = config.DOCKING_PARAMETERS.copy()

    # Проверяем, включен ли GPU режим
    if params.get("use_gpu", False):
        gpu_engine = params.get("gpu_engine", "vina_optimized")

        if gpu_engine == "vina_optimized":
            # Агрессивные настройки для GPU-подобной производительности
            params["num_threads"] = min(cpu_cores, 16)
            params["exhaustiveness"] = 64
            params["num_modes"] = 50
            params["energy_range"] = 5.0
            logger.info(f"GPU-оптимизированные параметры Vina: threads={params['num_threads']}, exhaustiveness={params['exhaustiveness']}, modes={params['num_modes']}")
        else:
            # Стандартные GPU параметры
            params["num_threads"] = cpu_cores // 2
            params["exhaustiveness"] = 32
            params["num_modes"] = 20
            params["energy_range"] = 4.0
            logger.info(f"GPU параметры Vina: threads={params['num_threads']}, exhaustiveness={params['exhaustiveness']}, modes={params['num_modes']}")
    else:
        # Стандартные CPU параметры
        if cpu_cores >= 16:
            params["num_threads"] = min(cpu_cores // 4, 8)
            params["exhaustiveness"] = 32
        elif cpu_cores >= 8:
            params["num_threads"] = cpu_cores // 2
            params["exhaustiveness"] = 16
        else:
            params["num_threads"] = max(1, cpu_cores // 2)
            params["exhaustiveness"] = 8

        params["num_modes"] = 20
        params["energy_range"] = 4.0
        logger.info(f"CPU параметры Vina: threads={params['num_threads']}, exhaustiveness={params['exhaustiveness']}, modes={params['num_modes']}")

    return params


def main() -> None:
    if not getattr(config, "USE_VINA_DOCKING", True):
        logger.info("USE_VINA_DOCKING flag is False – skipping AutoDock Vina execution.")
        return

    receptor = config.PROTEIN_PDBQT_PATH
    if not receptor.exists():
        logger.error(f"Receptor file {receptor} not found. Run protein_prep.py first.")
        _sys.exit(1)

    if not VINA_AVAILABLE:
        logger.warning("AutoDock Vina binary not found – will skip docking and return no scores.")
        _sys.exit(1)

    lig_files = sorted(config.LIGAND_PDBQT_DIR.glob("*.pdbqt"))
    # keep only original ligand files (exclude previously docked outputs)
    lig_files = [lf for lf in lig_files if not lf.stem.endswith("_dock")]
    # keep only files that have atoms (skip empty/invalid files)
    lig_files = [lf for lf in lig_files if has_atoms(lf)]

    if not lig_files:
        logger.error(f"No ligand PDBQT files found in {config.LIGAND_PDBQT_DIR}. Run ligand_prep.py first.")
        _sys.exit(1)

    logger.info(f"Docking {len(lig_files)} ligands with Vina…")

    # Определяем стратегию докинга на основе количества лигандов
    start_time = time.time()

    if len(lig_files) > 5000:
        # Для больших датасетов используем batch docking
        logger.info("Большой датасет - используем batch docking")
        batch_size = config.PIPELINE_PARAMETERS.get("docking", {}).get("batch_size", 1000)
        results = dock_ligands_batch(lig_files, batch_size)
    else:
        # Для средних датасетов используем параллельный докинг
        logger.info("Средний датасет - используем параллельный докинг")
        max_workers = config.PIPELINE_PARAMETERS.get("docking", {}).get("parallel_jobs", None)
        results = dock_ligands_parallel(lig_files, max_workers)

    elapsed_time = time.time() - start_time
    logger.info(f"Докинг завершен за {elapsed_time:.2f} секунд")

    if not results:
        logger.warning("No docking scores obtained.")
        return

    # Сохраняем финальные результаты
    df = pl.DataFrame(results, schema=["ligand_id", "docking_score"], orient="row")
    df.write_parquet(config.VINA_RESULTS_PATH)

    # Статистика
    avg_score = df["docking_score"].mean()
    min_score = df["docking_score"].min()
    max_score = df["docking_score"].max()

    logger.info(f"Docking completed. Scores saved to {config.VINA_RESULTS_PATH}")
    logger.info(f"Результаты: {len(results)} лигандов, среднее: {avg_score:.2f}, мин: {min_score:.2f}, макс: {max_score:.2f}")
    logger.info(f"Скорость: {len(results)/elapsed_time:.2f} лигандов/сек")


if __name__ == "__main__":
    main()
