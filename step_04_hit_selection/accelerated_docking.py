#!/usr/bin/env python3
"""Оптимизированный модуль для ускоренного молекулярного докинга
Поддерживает GPU-ускорение, параллелизацию и иерархический докинг
"""

import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path

import psutil
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# Добавляем путь к корневой директории проекта
sys.path.append(str(Path(__file__).parent.parent))

from config import *
from utils.logger import LOGGER as logger


@dataclass
class DockingJob:
    """Класс для представления задачи докинга"""

    ligand_id: str
    ligand_smiles: str
    ligand_pdbqt_path: str
    output_path: str
    priority: int = 0


class AcceleratedDocking:
    """Класс для ускоренного молекулярного докинга"""

    def __init__(self, config: dict | None = None):
        self.config = config if config is not None else DOCKING_PARAMETERS
        self.use_gpu = self.config.get("use_gpu", False) and self._check_vina_gpu()
        self.batch_size = self.config.get("batch_size", 1000)
        self.max_concurrent_jobs = self.config.get("max_concurrent_jobs", 4)
        self.timeout = self.config.get("timeout_per_ligand", 60)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="accelerated_docking_"))
        self.fast_screening_ratio = 0.1  # Доля молекул для точного докинга
        self.grid_generated = False  # Флаг, что сетка для AutoDock-GPU создана
        self.fld_path = None  # Путь к файлу сетки для AutoDock-GPU

        logger.info(f"Ускоренный докинг инициализирован. GPU: {self.use_gpu}")

    def _check_vina_gpu(self) -> bool:
        """Проверяет доступность GPU инструментов"""
        # Проверяем AutoDock-GPU
        autodock_gpu_path = self.config.get("autodock_gpu_path", "")
        if autodock_gpu_path and Path(autodock_gpu_path).exists():
            try:
                result = subprocess.run([autodock_gpu_path, "--help"], check=False, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    logger.info("AutoDock-GPU доступен")
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        # Проверяем Vina-GPU (для совместимости)
        vina_gpu_path = self.config.get("vina_gpu_path", "vina_gpu")
        if vina_gpu_path and Path(vina_gpu_path).exists():
            try:
                result = subprocess.run([vina_gpu_path, "--help"], check=False, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    logger.info("Vina-GPU доступен")
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        logger.warning("GPU инструменты не найдены, используется CPU Vina")
        return False

    def _run_vina_gpu_batch(self, ligand_dir: Path, output_dir: Path) -> dict[str, float]:
        """Запускает Vina-GPU для батча лигандов"""
        try:
            # Создаем конфигурационный файл для Vina-GPU
            config_file = self.temp_dir / "vina_gpu_config.txt"
            with open(config_file, "w") as f:
                f.write(f"receptor = {PROTEIN_PDBQT_PATH}\n")
                f.write(f"ligand_directory = {ligand_dir}\n")
                f.write(f"output_directory = {output_dir}\n")
                f.write(f"thread = {self.config.get('num_threads', 8000)}\n")
                f.write(f"center_x = {BOX_CENTER[0]}\n")
                f.write(f"center_y = {BOX_CENTER[1]}\n")
                f.write(f"center_z = {BOX_CENTER[2]}\n")
                f.write(f"size_x = {BOX_SIZE[0]}\n")
                f.write(f"size_y = {BOX_SIZE[1]}\n")
                f.write(f"size_z = {BOX_SIZE[2]}\n")
                f.write(f"exhaustiveness = {self.config.get('exhaustiveness', 8)}\n")
                f.write(f"num_modes = {self.config.get('num_modes', 9)}\n")
                f.write(f"energy_range = {self.config.get('energy_range', 3.0)}\n")

            # Запускаем Vina-GPU
            vina_gpu_path = self.config.get("vina_gpu_path", "vina_gpu")
            cmd = [vina_gpu_path, "--config", str(config_file)]

            start_time = time.time()
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, timeout=self.timeout * len(list(ligand_dir.glob("*.pdbqt")))
            )

            if result.returncode != 0:
                logger.error(f"Vina-GPU ошибка: {result.stderr}")
                return {}

            scores = self._parse_vina_gpu_results(output_dir)
            elapsed_time = time.time() - start_time
            logger.info(f"GPU-докинг завершен за {elapsed_time:.2f} сек для {len(scores)} лигандов")

            return scores

        except subprocess.TimeoutExpired:
            logger.error("GPU-докинг превысил таймаут")
            return {}
        except Exception as e:
            logger.error(f"Ошибка GPU-докинга: {e}")
            return {}

    def _run_vina_cpu_parallel(self, ligand_dir: Path, output_dir: Path) -> dict[str, float]:
        """Запускает параллельный CPU Vina"""
        scores = {}
        ligand_files = list(ligand_dir.glob("*.pdbqt"))

        with ThreadPoolExecutor(max_workers=self.max_concurrent_jobs) as executor:
            future_to_ligand = {}

            for ligand_file in ligand_files:
                future = executor.submit(self._dock_single_ligand, ligand_file, output_dir)
                future_to_ligand[future] = ligand_file

            for future in tqdm(as_completed(future_to_ligand), total=len(ligand_files), desc="CPU докинг"):
                ligand_file = future_to_ligand[future]
                try:
                    score = future.result()
                    if score is not None:
                        scores[ligand_file.stem] = score
                except Exception as e:
                    logger.error(f"Ошибка докинга {ligand_file}: {e}")

        return scores

    def _run_fast_cpu_docking(self, ligand_dir: Path, output_dir: Path) -> dict[str, float]:
        """Быстрый многопоточный CPU докинг"""
        ligand_files = list(ligand_dir.glob("*.pdbqt"))
        if not ligand_files:
            return {}

        scores = {}

        # Фильтруем только файлы с содержимым
        valid_ligands = []
        for ligand_file in ligand_files:
            if ligand_file.stat().st_size > 100:  # Минимальный размер файла
                valid_ligands.append(ligand_file)

        if not valid_ligands:
            logger.warning("Нет валидных лигандов для докинга")
            return {}

        # Многопоточный докинг
        max_workers = min(self.max_concurrent_jobs, len(valid_ligands))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ligand = {}

            for ligand_file in valid_ligands:
                future = executor.submit(self._dock_single_ligand, ligand_file, output_dir)
                future_to_ligand[future] = ligand_file

            # Собираем результаты с прогресс-баром
            for future in tqdm(as_completed(future_to_ligand), total=len(valid_ligands), desc="Быстрый CPU докинг"):
                ligand_file = future_to_ligand[future]
                try:
                    score = future.result()
                    if score is not None:
                        scores[ligand_file.stem] = score
                except Exception as e:
                    logger.error(f"Ошибка получения результата для {ligand_file}: {e}")

        return scores

    def _run_gpu_optimized_docking(self, ligand_dir: Path, output_dir: Path) -> dict[str, float]:
        """GPU-оптимизированный докинг с максимальным параллелизмом"""
        ligand_files = list(ligand_dir.glob("*.pdbqt"))
        if not ligand_files:
            return {}

        scores = {}

        # Фильтруем только файлы с содержимым
        valid_ligands = []
        for ligand_file in ligand_files:
            if ligand_file.stat().st_size > 100:  # Минимальный размер файла
                valid_ligands.append(ligand_file)

        if not valid_ligands:
            logger.warning("Нет валидных лигандов для докинга")
            return {}

        # Максимальный параллелизм с GPU поддержкой
        max_workers = min(32, len(valid_ligands))  # Используем больше потоков для GPU

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ligand = {}

            for ligand_file in valid_ligands:
                future = executor.submit(self._dock_single_ligand_gpu, ligand_file, output_dir)
                future_to_ligand[future] = ligand_file

            # Собираем результаты с прогресс-баром
            for future in tqdm(as_completed(future_to_ligand), total=len(valid_ligands), desc="GPU-оптимизированный докинг"):
                ligand_file = future_to_ligand[future]
                try:
                    score = future.result()
                    if score is not None:
                        scores[ligand_file.stem] = score
                except Exception as e:
                    logger.error(f"Ошибка получения результата для {ligand_file}: {e}")

        return scores

    def _run_autodock_gpu_batch(self, ligand_dir: Path, output_dir: Path) -> dict[str, float]:
        """Запускает AutoDock-GPU для батча лигандов с максимальной загрузкой GPU"""
        if not self.use_gpu:
            return {}

        # Создаем временную директорию для AutoDock-GPU
        output_dir.mkdir(exist_ok=True)

        ligand_files = list(ligand_dir.glob("*.pdbqt"))
        if not ligand_files:
            return {}

        start_time = time.time()
        scores = {}

        logger.info(f"Запуск AutoDock-GPU для {len(ligand_files)} лигандов с оптимизацией GPU")

        try:
            # Запускаем AutoDock-GPU для каждого лиганда
            autodock_gpu_path = self.config.get("autodock_gpu_path", "")

            # Разбиваем лиганды на более крупные батчи для лучшей загрузки GPU
            batch_size = min(self.config.get("batch_size", 2000), len(ligand_files))
            batches = [ligand_files[i : i + batch_size] for i in range(0, len(ligand_files), batch_size)]

            logger.info(f"Обработка {len(batches)} батчей по {batch_size} лигандов каждый")

            for batch_idx, batch_ligands in enumerate(batches):
                logger.info(f"Обработка батча {batch_idx + 1}/{len(batches)} ({len(batch_ligands)} лигандов)")

                # Используем максимальный параллелизм для GPU
                max_workers = min(self.config.get("max_concurrent_jobs", 16), len(batch_ligands))

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_ligand = {}

                    for ligand_file in batch_ligands:
                        ligand_name = ligand_file.stem
                        dlg_path = self.temp_dir / f"{ligand_name}.dlg"

                        # Оптимизированные параметры для максимальной загрузки GPU
                        cmd = [
                            autodock_gpu_path,
                            "--lfile",
                            str(ligand_file),
                            "--ffile",
                            str(self.fld_path),
                            "--resnam",
                            ligand_name,
                            "--devnum",
                            str(self.config.get("gpu_device", 0) + 1),
                            "--nrun",
                            str(self.config.get("autodock_gpu_nrun", 50)),
                            "--nev",
                            str(self.config.get("autodock_gpu_nev", 5000000)),
                            "--ngen",
                            str(self.config.get("autodock_gpu_ngen", 84000)),
                            "--psize",
                            str(self.config.get("autodock_gpu_psize", 300)),
                            "--heuristics",
                            str(self.config.get("autodock_gpu_heuristics", 1)),
                            "--autostop",
                            str(self.config.get("autodock_gpu_autostop", 0)),
                            "--dlgoutput",
                            "1",
                            "--xmloutput",
                            str(self.config.get("autodock_gpu_xml_output", 1)),
                        ]

                        future = executor.submit(self._run_single_autodock_gpu, cmd, ligand_name, dlg_path)
                        future_to_ligand[future] = ligand_name

                    # Собираем результаты с прогресс-баром
                    batch_scores = {}
                    completed = 0

                    for future in as_completed(future_to_ligand):
                        ligand_name = future_to_ligand[future]
                        try:
                            score = future.result()
                            if score is not None:
                                batch_scores[ligand_name] = score
                            completed += 1

                            # Логируем прогресс каждые 10% батча
                            if completed % max(1, len(batch_ligands) // 10) == 0:
                                logger.info(f"Батч {batch_idx + 1}: завершено {completed}/{len(batch_ligands)} лигандов")

                        except Exception as e:
                            logger.error(f"Ошибка получения результата для {ligand_name}: {e}")

                    scores.update(batch_scores)
                    logger.info(f"Батч {batch_idx + 1} завершен: {len(batch_scores)} успешных результатов")

            elapsed_time = time.time() - start_time
            logger.info(f"AutoDock-GPU докинг завершен за {elapsed_time:.2f} сек для {len(scores)} лигандов")
            logger.info(f"Производительность: {len(scores) / elapsed_time:.2f} лигандов/сек")

            return scores

        except Exception as e:
            logger.error(f"Критическая ошибка в AutoDock-GPU: {e}")
            return {}

    def _run_single_autodock_gpu(self, cmd: list[str], ligand_name: str, dlg_path: Path) -> float | None:
        """Запускает один процесс AutoDock-GPU"""
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.get("timeout_per_ligand", 600),
                cwd=str(self.temp_dir),
            )

            if result.returncode == 0:
                # Парсим результат из DLG файла
                if dlg_path.exists():
                    score = self._parse_autodock_gpu_dlg(dlg_path)
                    if score is not None:
                        return score

                # Если DLG файл не найден, пытаемся парсить из stdout
                score = self._parse_autodock_gpu_stdout(result.stdout)
                if score is not None:
                    return score

                logger.warning(f"Не удалось извлечь скор для {ligand_name}")
                return None
            logger.error(f"AutoDock-GPU ошибка для {ligand_name}: {result.stderr}")
            return None

        except subprocess.TimeoutExpired:
            logger.warning(f"Таймаут AutoDock-GPU для {ligand_name}")
            return None
        except Exception as e:
            logger.error(f"Исключение AutoDock-GPU для {ligand_name}: {e}")
            return None

    def _parse_autodock_gpu_dlg(self, dlg_path: Path) -> float | None:
        """Парсит DLG файл AutoDock-GPU для извлечения лучшего скора"""
        try:
            with open(dlg_path) as f:
                content = f.read()

            # Ищем строки с результатами кластеризации
            lines = content.split("\n")
            best_score = None

            for line in lines:
                # Ищем строки с энергией связывания
                if "DOCKED: USER    Final Intermolecular Energy" in line or "DOCKED: USER    Final Docked Energy" in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            score = float(parts[5])
                            if best_score is None or score < best_score:
                                best_score = score
                        except (ValueError, IndexError):
                            continue

            return best_score

        except Exception as e:
            logger.error(f"Ошибка парсинга DLG файла {dlg_path}: {e}")
            return None

    def _parse_autodock_gpu_stdout(self, stdout: str) -> float | None:
        """Парсит stdout AutoDock-GPU для извлечения скора"""
        try:
            lines = stdout.split("\n")

            for line in lines:
                # Ищем строки с результатами
                if "Best binding energy:" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "energy:" and i + 1 < len(parts):
                            try:
                                return float(parts[i + 1])
                            except ValueError:
                                continue

                # Альтернативный формат
                elif "Lowest binding energy:" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "energy:" in part and i + 1 < len(parts):
                            try:
                                return float(parts[i + 1])
                            except ValueError:
                                continue

            return None

        except Exception as e:
            logger.error(f"Ошибка парсинга stdout AutoDock-GPU: {e}")
            return None

    def _parse_autodock_gpu_result(self, dlg_path: Path) -> float | None:
        """Парсит результат AutoDock-GPU из DLG файла"""
        try:
            content = dlg_path.read_text()
            lines = content.split("\n")

            # Ищем различные форматы энергии в DLG файле
            for line in lines:
                line = line.strip()

                # Формат: DOCKED: ... ENERGY ... value
                if "DOCKED:" in line and "ENERGY" in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            return float(parts[3])
                        except ValueError:
                            continue

                # Формат: Lowest binding energy: value
                if "Lowest binding energy:" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            energy_str = parts[1].strip().split()[0]
                            return float(energy_str)
                        except (ValueError, IndexError):
                            continue

                # Формат: Final Docked Energy: value
                if "Final Docked Energy:" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            energy_str = parts[1].strip().split()[0]
                            return float(energy_str)
                        except (ValueError, IndexError):
                            continue

                # Формат табличный: номер энергия rmsd
                if line and line[0].isdigit():
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            # Второй элемент обычно энергия
                            return float(parts[1])
                        except ValueError:
                            continue

            # Если ничего не найдено, логируем первые несколько строк для отладки
            logger.debug("Не удалось найти энергию в DLG файле. Первые 10 строк:")
            for i, line in enumerate(lines[:10]):
                logger.debug(f"  {i + 1}: {line}")

            return None
        except Exception as e:
            logger.error(f"Ошибка парсинга DLG файла {dlg_path}: {e}")
            return None

    def _prepare_autodock_grid(self):
        """Подготавливает сетку для AutoDock-GPU"""
        try:
            # Создаем GPF файл для autogrid4
            gpf_path = self.temp_dir / "receptor.gpf"
            self._create_gpf_file(gpf_path)

            # Запускаем autogrid4 для создания сетки
            self._run_autogrid(gpf_path)

            # Устанавливаем путь к FLD файлу
            self.fld_path = self.temp_dir / "receptor.maps.fld"

            if self.fld_path.exists():
                self.grid_generated = True
                logger.info("Сетка для AutoDock-GPU успешно создана")
            else:
                logger.error("Не удалось создать сетку для AutoDock-GPU")

        except Exception as e:
            logger.error(f"Ошибка создания сетки для AutoDock-GPU: {e}")
            self.use_gpu = False

    def _create_gpf_file(self, gpf_path: Path):
        """Создает GPF файл для autogrid4"""
        try:
            # Конвертируем размер в ангстремах в количество точек сетки
            spacing = 0.375  # Стандартное расстояние между точками сетки в ангстремах
            npts_x = int(BOX_SIZE[0] / spacing)
            npts_y = int(BOX_SIZE[1] / spacing)
            npts_z = int(BOX_SIZE[2] / spacing)

            with open(gpf_path, "w") as f:
                f.write(f"npts {npts_x} {npts_y} {npts_z}\n")
                f.write("gridfld receptor.maps.fld\n")
                f.write(f"spacing {spacing}\n")
                f.write("receptor_types A C HD N NA OA S\n")  # Обновлено согласно реальным типам
                f.write("ligand_types A C HD N NA OA S\n")  # Обновлено согласно реальным типам
                f.write(f"receptor {PROTEIN_PDBQT_PATH}\n")
                f.write(f"gridcenter {BOX_CENTER[0]} {BOX_CENTER[1]} {BOX_CENTER[2]}\n")
                f.write("smooth 0.5\n")
                f.write("map receptor.A.map\n")
                f.write("map receptor.C.map\n")
                f.write("map receptor.HD.map\n")
                f.write("map receptor.N.map\n")
                f.write("map receptor.NA.map\n")  # Добавлено
                f.write("map receptor.OA.map\n")
                f.write("map receptor.S.map\n")  # Добавлено
                f.write("elecmap receptor.e.map\n")
                f.write("dsolvmap receptor.d.map\n")

        except Exception as e:
            logger.error(f"Ошибка создания GPF файла: {e}")
            raise

    def _run_autogrid(self, gpf_path: Path):
        """Запускает autogrid4 для создания сетки"""
        try:
            autogrid_path = self.config.get("autogrid_path", "/usr/local/bin/autogrid4")
            log_path = self.temp_dir / "autogrid.log"

            cmd = [autogrid_path, "-p", str(gpf_path), "-l", str(log_path)]

            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300, cwd=self.temp_dir)

            if result.returncode != 0:
                logger.error(f"autogrid4 завершился с ошибкой: {result.stderr}")
                raise RuntimeError(f"autogrid4 failed: {result.stderr}")

            logger.info("autogrid4 успешно завершен")

        except subprocess.TimeoutExpired:
            logger.error("autogrid4 превысил таймаут")
            raise
        except Exception as e:
            logger.error(f"Ошибка запуска autogrid4: {e}")
            raise

    def _dock_single_ligand(self, ligand_file: Path, output_dir: Path) -> float | None:
        """Докинг одного лиганда"""
        try:
            output_file = output_dir / f"{ligand_file.stem}_out.pdbqt"

            cmd = [
                "vina",
                "--receptor",
                str(PROTEIN_PDBQT_PATH),
                "--ligand",
                str(ligand_file),
                "--out",
                str(output_file),
                "--center_x",
                str(BOX_CENTER[0]),
                "--center_y",
                str(BOX_CENTER[1]),
                "--center_z",
                str(BOX_CENTER[2]),
                "--size_x",
                str(BOX_SIZE[0]),
                "--size_y",
                str(BOX_SIZE[1]),
                "--size_z",
                str(BOX_SIZE[2]),
                "--exhaustiveness",
                str(self.config.get("exhaustiveness", 8)),
                "--num_modes",
                str(self.config.get("num_modes", 9)),
                "--energy_range",
                str(self.config.get("energy_range", 3.0)),
                "--cpu",
                str(self.config.get("num_threads", cpu_count())),
                "--verbosity",
                "1",  # Минимальный вывод с результатами
            ]

            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=self.timeout)

            if result.returncode == 0:
                score = self._parse_vina_output(result.stdout)
                if score is not None:
                    return score
                # Пытаемся прочитать скор из выходного файла
                if output_file.exists():
                    try:
                        with open(output_file) as f:
                            content = f.read()
                            # Ищем скор в REMARK строках
                            for line in content.split("\n"):
                                if "REMARK VINA RESULT:" in line:
                                    parts = line.split()
                                    if len(parts) >= 4:
                                        return float(parts[3])
                    except Exception as e:
                        logger.error(f"Ошибка чтения файла результата: {e}")

                logger.warning(f"Не удалось извлечь скор из вывода Vina для {ligand_file}")
                return None
            logger.error(f"Vina ошибка для {ligand_file} (код {result.returncode})")
            logger.error(f"stderr: {result.stderr}")
            return None

        except subprocess.TimeoutExpired:
            logger.warning(f"Таймаут для {ligand_file}")
            return None
        except Exception as e:
            logger.error(f"Ошибка докинга {ligand_file}: {e}")
            return None

    def _dock_single_ligand_gpu(self, ligand_file: Path, output_dir: Path) -> float | None:
        """Докинг одного лиганда с GPU оптимизацией"""
        try:
            # Используем Vina с GPU-оптимизированными настройками
            cmd = [
                "vina",
                "--receptor",
                str(PROTEIN_PDBQT_PATH),
                "--ligand",
                str(ligand_file),
                "--center_x",
                str(BOX_CENTER[0]),
                "--center_y",
                str(BOX_CENTER[1]),
                "--center_z",
                str(BOX_CENTER[2]),
                "--size_x",
                str(BOX_SIZE[0]),
                "--size_y",
                str(BOX_SIZE[1]),
                "--size_z",
                str(BOX_SIZE[2]),
                "--cpu",
                "4",  # Используем больше CPU cores для GPU систем
                "--exhaustiveness",
                "16",  # Увеличиваем exhaustiveness для GPU
                "--num_modes",
                "1",  # Только лучший режим для скорости
                "--energy_range",
                "3",  # Сокращаем energy range для скорости
                "--verbosity",
                "0",  # Тихий режим
                "--out",
                str(output_dir / f"{ligand_file.stem}_out.pdbqt"),
            ]

            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                score = self._parse_vina_output(result.stdout)
                return score
            logger.debug(f"Vina GPU ошибка для {ligand_file.stem}: {result.stderr}")
            return None

        except subprocess.TimeoutExpired:
            logger.debug(f"Timeout для {ligand_file.stem}")
            return None
        except Exception as e:
            logger.debug(f"Ошибка докинга {ligand_file.stem}: {e}")
            return None

    def _parse_vina_gpu_results(self, output_dir: Path) -> dict[str, float]:
        """Парсит результаты Vina-GPU"""
        scores = {}

        for result_file in output_dir.glob("*.pdbqt"):
            try:
                with open(result_file) as f:
                    content = f.read()
                    score = self._parse_vina_output(content)
                    if score is not None:
                        scores[result_file.stem.replace("_out", "")] = score
            except Exception as e:
                logger.error(f"Ошибка парсинга {result_file}: {e}")

        return scores

    def _parse_vina_output(self, output: str) -> float | None:
        """Парсит вывод Vina для извлечения лучшего скора"""
        try:
            lines = output.split("\n")
            for line in lines:
                if "REMARK VINA RESULT:" in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        return float(parts[3])
            return None
        except Exception as e:
            logger.error(f"Ошибка парсинга скора: {e}")
            return None

    def dock_molecules_batch(self, molecules: list[dict]) -> dict[str, float]:
        """Основная функция для батч-докинга молекул"""
        logger.info(f"Начинаем ускоренный докинг {len(molecules)} молекул")

        all_scores = {}

        for i in range(0, len(molecules), self.batch_size):
            batch = molecules[i : i + self.batch_size]
            logger.info(f"Обрабатываем батч {i // self.batch_size + 1}: {len(batch)} молекул")

            # Создаем временные директории для батча
            batch_ligand_dir = self.temp_dir / f"batch_{i // self.batch_size}_ligands"
            batch_output_dir = self.temp_dir / f"batch_{i // self.batch_size}_outputs"
            batch_ligand_dir.mkdir(exist_ok=True)
            batch_output_dir.mkdir(exist_ok=True)

            # Подготавливаем лиганды
            ligand_files = []
            for mol in batch:
                ligand_file = batch_ligand_dir / f"{mol.get('id', 'mol')}.pdbqt"
                try:
                    success = self._prepare_ligand_stub(mol, ligand_file)
                    if success:
                        ligand_files.append(ligand_file)
                except Exception as e:
                    logger.error(f"Ошибка подготовки лиганда {mol.get('id', 'unknown')}: {e}")

            # Запускаем докинг для батча
            if ligand_files:
                if self.use_gpu:
                    gpu_engine = self.config.get("gpu_engine", "autodock_gpu")
                    if gpu_engine == "autodock_gpu":
                        logger.info("Используем AutoDock-GPU для ускоренного докинга")
                        # Подготавливаем сетку для AutoDock-GPU если еще не готова
                        if not self.grid_generated:
                            self._prepare_autodock_grid()
                        batch_scores = self._run_autodock_gpu_batch(batch_ligand_dir, batch_output_dir)
                    else:
                        logger.info("Используем GPU-оптимизированный докинг")
                        batch_scores = self._run_gpu_optimized_docking(batch_ligand_dir, batch_output_dir)
                else:
                    logger.info("Используем быстрый многопоточный CPU докинг")
                    batch_scores = self._run_fast_cpu_docking(batch_ligand_dir, batch_output_dir)
                all_scores.update(batch_scores)

            # Очищаем временные файлы батча
            if batch_ligand_dir.exists():
                shutil.rmtree(batch_ligand_dir)
            if batch_output_dir.exists():
                shutil.rmtree(batch_output_dir)

        logger.info(f"Ускоренный докинг завершен. Получено {len(all_scores)} результатов")
        return all_scores

    def _prepare_ligand_stub(self, mol: dict, output_file: Path) -> bool:
        """Полноценная функция подготовки лиганда: SMILES -> 3D PDB -> PDBQT.
        Заменяет старую заглушку.
        """
        smiles = mol.get("smiles") or mol.get("SMILES")
        mol_id = mol.get("id", "unknown_mol")

        if not smiles:
            logger.warning(f"[{mol_id}] Пропущено: отсутствует SMILES.")
            return False

        # --- Этап 1: Генерация 3D-структуры (PDB) с помощью RDKit ---
        try:
            rdkit_mol = Chem.MolFromSmiles(smiles)
            if not rdkit_mol:
                logger.warning(f"[{mol_id}] RDKit не смог прочитать SMILES: {smiles}")
                return False

            rdkit_mol = Chem.AddHs(rdkit_mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xF00D  # Для воспроизводимости
            if AllChem.EmbedMolecule(rdkit_mol, params) == -1:
                logger.warning(f"[{mol_id}] RDKit не смог сгенерировать 3D-конформер для {smiles}")
                return False

            AllChem.UFFOptimizeMolecule(rdkit_mol)

            # Используем временный PDB файл
            temp_pdb_path = output_file.with_suffix(".tmp.pdb")
            Chem.MolToPDBFile(rdkit_mol, str(temp_pdb_path))

        except Exception as e:
            logger.error(f"[{mol_id}] Критическая ошибка на этапе RDKit для {smiles}: {e}")
            return False

        # --- Этап 2: Конвертация PDB в PDBQT с помощью OpenBabel ---
        cmd = ["obabel", str(temp_pdb_path), "-O", str(output_file), "--partialcharge", "gasteiger"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            error_msg = e.stderr.strip() if hasattr(e, "stderr") else str(e)
            logger.error(f"[{mol_id}] OpenBabel не смог конвертировать {smiles}. Ошибка: {error_msg}")
            temp_pdb_path.unlink(missing_ok=True)
            return False
        finally:
            # Гарантированно удаляем временный файл
            if temp_pdb_path.exists():
                temp_pdb_path.unlink()

        # --- Этап 3: Валидация ---
        if not output_file.exists() or output_file.stat().st_size == 0:
            logger.warning(f"[{mol_id}] OpenBabel создал пустой PDBQT файл для {smiles}.")
            return False

        return True

    def __del__(self):
        """Очистка при удалении объекта"""
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


class HierarchicalDocking:
    """Иерархический докинг: быстрый скрининг + точный докинг"""

    def __init__(self, config: dict | None = None):
        self.config = config or DOCKING_PARAMETERS
        self.fast_screening_ratio = 0.1  # Доля молекул для точного докинга

    def dock_molecules(self, molecules: list[dict]) -> dict[str, float]:
        """Иерархический докинг молекул"""
        logger.info(f"Начинаем иерархический докинг {len(molecules)} молекул")

        # Этап 1: Быстрый скрининг
        logger.info("Этап 1: Быстрый скрининг")
        fast_config = self.config.copy()
        fast_config["exhaustiveness"] = 4
        fast_config["num_modes"] = 3
        fast_config["timeout_per_ligand"] = 30

        fast_docking = AcceleratedDocking(fast_config)
        fast_scores = fast_docking.dock_molecules_batch(molecules)

        # Этап 2: Точный докинг топ молекул
        if fast_scores:
            sorted_scores = sorted(fast_scores.items(), key=lambda x: x[1])
            top_count = max(1, int(len(sorted_scores) * self.fast_screening_ratio))
            top_molecules = []

            mol_dict = {mol.get("id", f"mol_{i}"): mol for i, mol in enumerate(molecules)}
            for mol_id, score in sorted_scores[:top_count]:
                if mol_id in mol_dict:
                    top_molecules.append(mol_dict[mol_id])

            logger.info(f"Этап 2: Точный докинг топ {len(top_molecules)} молекул")

            # Точный докинг
            precise_docking = AcceleratedDocking(self.config)
            precise_scores = precise_docking.dock_molecules_batch(top_molecules)

            # Комбинируем результаты
            final_scores = fast_scores.copy()
            final_scores.update(precise_scores)

            return final_scores

        return fast_scores


def optimize_docking_performance() -> dict:
    """Оптимизирует параметры докинга на основе системных ресурсов"""
    logger.info("Оптимизируем параметры докинга")

    # Проверяем доступные ресурсы
    cpu_count_val = cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)

    logger.info(f"Доступно CPU: {cpu_count_val}, память: {memory_gb:.1f} GB")

    # Оптимизируем параметры
    optimal_config = DOCKING_PARAMETERS.copy()

    # Автоматическая настройка количества потоков
    optimal_config["num_threads"] = min(cpu_count_val, 16)
    optimal_config["max_concurrent_jobs"] = min(cpu_count_val // 2, 8)

    # Настройка размера батча на основе доступной памяти
    if memory_gb > 32:
        optimal_config["batch_size"] = 2000
    elif memory_gb > 16:
        optimal_config["batch_size"] = 1000
    else:
        optimal_config["batch_size"] = 500

    return optimal_config


def run_accelerated_docking_demo():
    """Демонстрация ускоренного докинга"""
    logger.info("=== Демонстрация ускоренного докинга ===")

    # Оптимизируем параметры
    optimal_config = optimize_docking_performance()

    # Создаем тестовые молекулы
    test_molecules = [{"id": f"mol_{i}", "smiles": f"C{'C' * i}N"} for i in range(10)]

    # Обычный докинг
    logger.info("Тестируем обычный ускоренный докинг...")
    start_time = time.time()

    docking = AcceleratedDocking(optimal_config)
    scores = docking.dock_molecules_batch(test_molecules)

    normal_time = time.time() - start_time
    logger.info(f"Обычный докинг: {normal_time:.2f} сек, {len(scores)} результатов")

    # Иерархический докинг
    logger.info("Тестируем иерархический докинг...")
    start_time = time.time()

    hierarchical = HierarchicalDocking(optimal_config)
    hierarchical_scores = hierarchical.dock_molecules(test_molecules)

    hierarchical_time = time.time() - start_time
    logger.info(f"Иерархический докинг: {hierarchical_time:.2f} сек, {len(hierarchical_scores)} результатов")

    # Сравнение результатов
    speedup = normal_time / hierarchical_time if hierarchical_time > 0 else 1
    logger.info(f"Ускорение: {speedup:.2f}x")

    return scores, hierarchical_scores


if __name__ == "__main__":
    run_accelerated_docking_demo()
