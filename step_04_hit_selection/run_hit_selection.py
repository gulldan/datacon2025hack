# step_04_hit_selection/run_hit_selection.py
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
from tqdm import tqdm

# Попытка импорта GPUtil с fallback
try:
    import GPUtil

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    # logger будет инициализирован позже

# Добавляем путь к корневой директории проекта
sys.path.append(str(Path(__file__).parent.parent))

from config import *
from utils.logger import LOGGER as logger

# Логируем предупреждение о GPUtil после инициализации logger
if not GPU_AVAILABLE:
    logger.warning("GPUtil не установлен. GPU-мониторинг недоступен.")


@dataclass
class DockingJob:
    """Класс для представления задачи докинга"""

    ligand_id: str
    ligand_smiles: str
    ligand_pdbqt_path: str
    output_path: str
    priority: int = 0  # Приоритет задачи (0 - высший)


class GPUAcceleratedDocking:
    """Класс для GPU-ускоренного докинга"""

    def __init__(self, config: dict):
        self.config = config
        self.vina_gpu_path = config.get("vina_gpu_path", "/usr/local/bin/vina_gpu")
        self.gpu_device = config.get("gpu_device", 0)
        self.batch_size = config.get("batch_size", 1000)
        self.max_concurrent_jobs = config.get("max_concurrent_jobs", 4)
        self.timeout = config.get("timeout_per_ligand", 60)
        self.use_gpu = config.get("use_gpu", True) and self._check_gpu_availability()

        # Создаем временную директорию для GPU-докинга
        self.temp_dir = Path(tempfile.mkdtemp(prefix="gpu_docking_"))

        logger.info(f"GPU-ускоренный докинг инициализирован. GPU доступен: {self.use_gpu}")

    def _check_gpu_availability(self) -> bool:
        """Проверяет доступность GPU"""
        if not GPU_AVAILABLE:
            logger.warning("GPUtil не доступен")
            return False

        try:
            gpus = GPUtil.getGPUs()  # type: ignore
            if gpus:
                gpu = gpus[self.gpu_device] if self.gpu_device < len(gpus) else gpus[0]
                logger.info(f"Найден GPU: {gpu.name}, память: {gpu.memoryFree}MB свободно")
                return True
            logger.warning("GPU не найден, используется CPU")
            return False
        except Exception as e:
            logger.warning(f"Ошибка при проверке GPU: {e}")
            return False

    def _run_vina_gpu_batch(self, ligand_dir: Path, output_dir: Path) -> dict[str, float]:
        """Запускает Vina-GPU для батча лигандов"""
        if not self.use_gpu:
            return self._run_vina_cpu_batch(ligand_dir, output_dir)

        try:
            # Создаем конфигурационный файл для Vina-GPU
            config_file = self.temp_dir / "vina_gpu_config.txt"
            with open(config_file, "w") as f:
                f.write(f"receptor = {PROTEIN_PDBQT_PATH}\n")
                f.write(f"ligand_directory = {ligand_dir}\n")
                f.write(f"output_directory = {output_dir}\n")
                f.write(f"thread = {self.config.get('num_threads', 8000)}\n")
                f.write(f"opencl_binary_path = {self.config.get('opencl_binary_path', '')}\n")
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
            cmd = [self.vina_gpu_path, "--config", str(config_file)]

            start_time = time.time()
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, timeout=self.timeout * len(list(ligand_dir.glob("*.pdbqt")))
            )

            if result.returncode != 0:
                logger.error(f"Vina-GPU ошибка: {result.stderr}")
                return {}

            # Парсим результаты
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

    def _run_vina_cpu_batch(self, ligand_dir: Path, output_dir: Path) -> dict[str, float]:
        """Запускает обычный Vina для батча лигандов"""
        scores = {}
        ligand_files = list(ligand_dir.glob("*.pdbqt"))

        # Параллельная обработка с помощью ThreadPoolExecutor
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
            ]

            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=self.timeout)

            if result.returncode == 0:
                return self._parse_vina_output(result.stdout)
            logger.error(f"Vina ошибка для {ligand_file}: {result.stderr}")
            return None

        except subprocess.TimeoutExpired:
            logger.warning(f"Таймаут для {ligand_file}")
            return None
        except Exception as e:
            logger.error(f"Ошибка докинга {ligand_file}: {e}")
            return None

    def _parse_vina_gpu_results(self, output_dir: Path) -> dict[str, float]:
        """Парсит результаты Vina-GPU"""
        scores = {}

        # Ищем файлы результатов
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

    def dock_molecules(self, molecules: list[dict]) -> dict[str, float]:
        """Основная функция для докинга молекул"""
        logger.info(f"Начинаем GPU-ускоренный докинг {len(molecules)} молекул")

        # Подготавливаем лиганды батчами
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
                ligand_file = batch_ligand_dir / f"{mol['id']}.pdbqt"
                try:
                    # Здесь должна быть функция конвертации SMILES в PDBQT
                    success = self._prepare_ligand(mol, ligand_file)
                    if success:
                        ligand_files.append(ligand_file)
                except Exception as e:
                    logger.error(f"Ошибка подготовки лиганда {mol['id']}: {e}")

            # Запускаем докинг для батча
            if ligand_files:
                batch_scores = self._run_vina_gpu_batch(batch_ligand_dir, batch_output_dir)
                all_scores.update(batch_scores)

            # Очищаем временные файлы батча
            if batch_ligand_dir.exists():
                shutil.rmtree(batch_ligand_dir)
            if batch_output_dir.exists():
                shutil.rmtree(batch_output_dir)

        # Очищаем временную директорию
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

        logger.info(f"GPU-докинг завершен. Получено {len(all_scores)} результатов")
        return all_scores

    def __del__(self):
        """Очистка при удалении объекта"""
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _prepare_ligand(self, mol_data: dict, output_pdbqt_path: Path) -> bool:
        """ОТЛАДОЧНАЯ ВЕРСИЯ функции подготовки лиганда."""
        smiles = mol_data.get("smiles")
        mol_id = mol_data.get("id", "unknown_mol")
        logger.info(f"--- [{mol_id}] НАЧАЛО ПОДГОТОВКИ для SMILES: {smiles} ---")

        if not smiles:
            logger.error(f"[{mol_id}] СБОЙ: отсутствует SMILES.")
            return False

        # --- Этап 1: RDKit ---
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            logger.error(f"[{mol_id}] СБОЙ: RDKit не смог прочитать SMILES.")
            return False
        logger.info(f"[{mol_id}] RDKit: SMILES успешно прочитан.")

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        if AllChem.EmbedMolecule(mol, params) == -1:
            logger.error(f"[{mol_id}] СБОЙ: RDKit не смог сгенерировать 3D-конформер.")
            return False
        logger.info(f"[{mol_id}] RDKit: 3D-конформер успешно сгенерирован.")

        try:
            AllChem.UFFOptimizeMolecule(mol)
            logger.info(f"[{mol_id}] RDKit: 3D-структура успешно оптимизирована.")
        except Exception as e:
            logger.error(f"[{mol_id}] СБОЙ: Ошибка оптимизации 3D: {e}")
            return False

        temp_pdb_path = output_pdbqt_path.with_suffix(".tmp.pdb")
        Chem.MolToPDBFile(mol, str(temp_pdb_path))
        logger.info(f"[{mol_id}] RDKit: Временный PDB файл сохранен в {temp_pdb_path}.")

        # --- Этап 2: OpenBabel ---
        cmd = ["obabel", str(temp_pdb_path), "-O", str(output_pdbqt_path), "--partialcharge", "gasteiger"]
        logger.info(f"[{mol_id}] OpenBabel: Запуск команды: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
            logger.info(f"[{mol_id}] OpenBabel: Команда успешно выполнена.")
        except Exception as e:
            error_msg = e.stderr.strip() if hasattr(e, "stderr") else str(e)
            logger.error(f"[{mol_id}] СБОЙ: OpenBabel не смог конвертировать. Ошибка: {error_msg}")
            temp_pdb_path.unlink(missing_ok=True)
            return False
        finally:
            temp_pdb_path.unlink(missing_ok=True)

        # --- Этап 3: Валидация ---
        if not output_pdbqt_path.exists() or output_pdbqt_path.stat().st_size == 0:
            logger.error(f"[{mol_id}] СБОЙ: OpenBabel создал пустой PDBQT файл.")
            return False

        logger.info(f"--- [{mol_id}] УСПЕХ: PDBQT файл готов: {output_pdbqt_path} ---")
        return True


class HierarchicalDocking:
    """Иерархический докинг: быстрый скрининг + точный докинг"""

    def __init__(self, config: dict):
        self.config = config
        self.gpu_docking = GPUAcceleratedDocking(config)
        self.fast_screening_ratio = 0.1  # Доля молекул для точного докинга

    def dock_molecules(self, molecules: list[dict]) -> dict[str, float]:
        """Иерархический докинг молекул"""
        logger.info(f"Начинаем иерархический докинг {len(molecules)} молекул")

        # Этап 1: Быстрый скрининг всех молекул
        logger.info("Этап 1: Быстрый скрининг")
        fast_config = self.config.copy()
        fast_config["exhaustiveness"] = 4  # Снижаем точность для скорости
        fast_config["num_modes"] = 3
        fast_config["timeout_per_ligand"] = 30

        fast_docking = GPUAcceleratedDocking(fast_config)
        fast_scores = fast_docking.dock_molecules(molecules)

        # Этап 2: Отбираем топ молекул для точного докинга
        if fast_scores:
            sorted_scores = sorted(fast_scores.items(), key=lambda x: x[1])
            top_count = max(1, int(len(sorted_scores) * self.fast_screening_ratio))
            top_molecules = []

            mol_dict = {mol["id"]: mol for mol in molecules}
            for mol_id, score in sorted_scores[:top_count]:
                if mol_id in mol_dict:
                    top_molecules.append(mol_dict[mol_id])

            logger.info(f"Этап 2: Точный докинг топ {len(top_molecules)} молекул")

            # Точный докинг с оригинальными параметрами
            precise_scores = self.gpu_docking.dock_molecules(top_molecules)

            # Комбинируем результаты
            final_scores = fast_scores.copy()
            final_scores.update(precise_scores)

            return final_scores

        return fast_scores


def optimize_docking_performance():
    """Оптимизирует производительность системы для докинга"""
    logger.info("Оптимизируем производительность системы")

    # Проверяем доступные ресурсы
    cpu_count_val = cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)

    logger.info(f"Доступно CPU: {cpu_count_val}, память: {memory_gb:.1f} GB")

    # Проверяем GPU
    try:
        if GPU_AVAILABLE:
            gpus = GPUtil.getGPUs()  # type: ignore
            if gpus:
                for i, gpu in enumerate(gpus):
                    logger.info(f"GPU {i}: {gpu.name}, память: {gpu.memoryTotal}MB")
        else:
            logger.info("GPU не обнаружен или GPUtil не установлен")
    except Exception as e:
        logger.info(f"Ошибка при проверке GPU: {e}")

    # Оптимизируем параметры на основе доступных ресурсов
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


def run_hit_selection_pipeline():
    """Основная функция для запуска пайплайна отбора хитов"""
    logger.info("🎯 Запуск пайплайна отбора хитов (Hit Selection)")

    try:
        # Импортируем необходимые модули
        import sys
        from pathlib import Path

        import polars as pl

        sys.path.append(str(Path(__file__).parent.parent))
        from config import GENERATED_MOLECULES_PATH, HIT_SELECTION_RESULTS_DIR

        # Проверяем наличие сгенерированных молекул
        if not Path(GENERATED_MOLECULES_PATH).exists():
            logger.error(f"Файл с сгенерированными молекулами не найден: {GENERATED_MOLECULES_PATH}")
            logger.info("Сначала запустите этап 3: Генерация молекул")
            return

        # Загружаем сгенерированные молекулы
        logger.info("📄 Загрузка сгенерированных молекул")
        df = pl.read_parquet(GENERATED_MOLECULES_PATH)
        molecules = df.to_dicts()
        logger.info(f"Загружено {len(molecules)} молекул")

        # Оптимизируем производительность
        logger.info("⚡ Оптимизация производительности системы")
        optimal_config = optimize_docking_performance()

        # Инициализируем ускоренный докинг
        logger.info("🚀 Инициализация ускоренного докинга")
        from step_04_hit_selection.accelerated_docking import AcceleratedDocking

        docking_engine = AcceleratedDocking(optimal_config)

        # Запускаем докинг
        logger.info("🎯 Запуск молекулярного докинга")

        # Ограничиваем количество молекул для демонстрации
        demo_molecules = molecules
        logger.info(f"Обрабатываем {len(demo_molecules)} молекул для демонстрации")

        # Добавляем необходимые поля если их нет
        for i, mol in enumerate(demo_molecules):
            if "id" not in mol:
                mol["id"] = f"mol_{i}"
            if "smiles" not in mol and "SMILES" in mol:
                mol["smiles"] = mol["SMILES"]

        # Выполняем докинг
        scores = docking_engine.dock_molecules_batch(demo_molecules)

        if scores:
            logger.info(f"✅ Докинг завершен успешно! Получено {len(scores)} результатов")

            # Сохраняем результаты
            results_dir = Path(HIT_SELECTION_RESULTS_DIR)
            results_dir.mkdir(parents=True, exist_ok=True)

            # Преобразуем результаты в DataFrame
            results_data = []
            for mol_id, score in scores.items():
                mol_data = next((m for m in demo_molecules if m.get("id") == mol_id), {})
                results_data.append(
                    {
                        "molecule_id": mol_id,
                        "smiles": mol_data.get("smiles", ""),
                        "docking_score": score,
                        "rank": 0,  # Будет заполнено после сортировки
                    }
                )

            # Сортируем по скору докинга (лучшие = более отрицательные)
            results_data.sort(key=lambda x: x["docking_score"])
            for i, result in enumerate(results_data):
                result["rank"] = i + 1

            # Сохраняем результаты
            results_df = pl.DataFrame(results_data)
            output_path = results_dir / "final_hits.parquet"
            results_df.write_parquet(output_path)
            logger.info(f"💾 Результаты сохранены в: {output_path}")

            # Выводим топ результаты
            logger.info("🏆 Топ-10 результатов:")
            for i, result in enumerate(results_data[:10]):
                logger.info(f"  {i + 1}. {result['molecule_id']}: {result['docking_score']:.3f}")

        else:
            logger.warning("⚠️ Докинг не дал результатов")

    except Exception as e:
        logger.error(f"❌ Ошибка в пайплайне отбора хитов: {e}")
        raise

    logger.info("🎉 Пайплайн отбора хитов завершен")
