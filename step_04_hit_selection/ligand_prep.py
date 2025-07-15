"""Ligand preparation for AutoDock Vina.

Reads `generated_molecules.parquet`, generates 3D conformers with RDKit ETKDG,
adds hydrogens and converts to PDBQT via OpenBabel CLI.

This optimized version processes ligands in parallel to significantly speed up execution.
"""

from __future__ import annotations

import subprocess
import sys as _sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# --- Конфигурация (предполагается, что эти переменные определены в config.py) ---

# Добавляем корневую директорию в путь для импорта
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in _sys.path:
    _sys.path.insert(0, str(ROOT_DIR))

# Импортируем конфигурацию и логгер из вашего проекта
# Замените на ваши реальные импорты, если они отличаются
try:
    import config
    from utils.logger import LOGGER
except ImportError:
    # Заглушки, если скрипт запускается отдельно
    class MockConfig:
        GENERATED_MOLECULES_PATH = Path("generated_molecules.parquet")
        LIGAND_PDBQT_DIR = Path("ligands_pdbqt")

    config = MockConfig()
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    LOGGER = logging.getLogger(__name__)

# --- Существующие функции (сигнатуры не изменены) ---


def smiles_to_3d_pdb(smiles: str, out_path: Path) -> bool:
    """Генерирует 3D-конформер из SMILES и сохраняет в формате PDB.
    Улучшенная версия с явной проверкой результата генерации конформера.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        LOGGER.warning(f"RDKit не смог прочитать SMILES: {smiles}")
        return False

    mol = Chem.AddHs(mol)

    try:
        # ETKDG v3 - улучшенный алгоритм генерации конформеров
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D  # для воспроизводимости

        # Шаг 1: Генерация конформера
        embed_result = AllChem.EmbedMolecule(mol, params)

        # ЯВНАЯ ПРОВЕРКА: EmbedMolecule возвращает -1 при неудаче
        if embed_result == -1:
            LOGGER.error(f"Не удалось найти конформер для {smiles}. Молекула слишком стерически напряжена.")
            return False

        # Шаг 2: Оптимизация, только если конформер был успешно создан
        AllChem.UFFOptimizeMolecule(mol)

    except Exception as e:
        # Этот блок теперь будет ловить другие, менее ожидаемые ошибки
        LOGGER.error(f"Произошла непредвиденная ошибка при генерации 3D для {smiles}: {e}")
        return False

    Chem.MolToPDBFile(mol, str(out_path))
    return True


def pdb_to_pdbqt(pdb_path: Path, pdbqt_path: Path) -> None:
    """Конвертирует PDB в PDBQT с помощью OpenBabel и выполняет постобработку."""
    cmd = ["obabel", str(pdb_path), "-O", str(pdbqt_path), "--partialcharge", "gasteiger"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if pdbqt_path.exists():
            fix_pdbqt_multiple_roots(pdbqt_path)
    except subprocess.CalledProcessError as e:
        # Логируем stderr от OpenBabel для лучшей диагностики
        LOGGER.warning(f"Ошибка конвертации в OpenBabel для {pdb_path.name}: {e.stderr.strip()}")


def fix_pdbqt_multiple_roots(pdbqt_path: Path) -> None:
    """Исправляет PDBQT файлы с несколькими секциями ROOT, оставляя только первую."""
    try:
        with open(pdbqt_path) as f:
            lines = f.readlines()

        root_indices = [i for i, line in enumerate(lines) if line.strip() == "ROOT"]

        # Если секций ROOT больше одной, обрезаем файл после первой ENDROOT
        if len(root_indices) > 1:
            try:
                endroot_index = lines.index("ENDROOT\n", root_indices[0])
                lines = lines[: endroot_index + 1]
            except ValueError:
                LOGGER.warning(f"Найдены несколько ROOT, но не найден ENDROOT в {pdbqt_path}. Файл может быть поврежден.")
                return

        # Добавляем TORSDOF, если его нет (важно для жестких лигандов)
        if not any("TORSDOF" in line for line in lines):
            # Вставляем перед последней строкой, если это ENDROOT/BRANCH, иначе добавляем в конец
            insert_pos = len(lines)
            if lines and lines[-1].strip().startswith(("ENDROOT", "ENDBRANCH")):
                insert_pos = -1
            lines.insert(insert_pos, "TORSDOF 0\n")

        with open(pdbqt_path, "w") as f:
            f.writelines(lines)

    except Exception as e:
        LOGGER.warning(f"Не удалось исправить PDBQT файл {pdbqt_path}: {e}")


def is_valid_pdbqt(pdbqt_path: Path) -> bool:
    """Возвращает True, если PDBQT файл существует и содержит хотя бы одну строку ATOM/HETATM."""
    if not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
        return False
    try:
        with open(pdbqt_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    return True
    except Exception as e:
        LOGGER.error(f"Ошибка при чтении {pdbqt_path}: {e}")
        return False
    return False


# --- Новая "рабочая" функция для параллельной обработки ---


def _prepare_single(args: tuple[int, str], ligand_dir: Path) -> bool:
    """Выполняет полный цикл подготовки для одного лиганда.
    Предназначена для использования с multiprocessing.Pool.
    """
    idx, smi = args

    # Пропускаем уже обработанные лиганды
    pdbqt_path = ligand_dir / f"lig_{idx}.pdbqt"
    if pdbqt_path.exists():
        return True  # Считаем успешным, если файл уже есть

    pdb_path = ligand_dir / f"lig_{idx}.pdb"

    # Шаг 1: SMILES -> 3D PDB
    if not smiles_to_3d_pdb(smi, pdb_path):
        LOGGER.warning(f"[{idx}] Пропуск лиганда: не удалось сгенерировать 3D для SMILES: {smi}")
        # Удаляем пустой PDB, если он был создан
        pdb_path.unlink(missing_ok=True)
        return False

    # Шаг 2: PDB -> PDBQT
    pdb_to_pdbqt(pdb_path, pdbqt_path)

    # Шаг 3: Валидация и очистка
    if not is_valid_pdbqt(pdbqt_path):
        LOGGER.warning(f"[{idx}] OpenBabel не смог создать валидный PDBQT для {smi}. Пропуск лиганда.")
        pdbqt_path.unlink(missing_ok=True)  # Удаляем невалидный PDBQT
        pdb_path.unlink(missing_ok=True)  # Удаляем промежуточный PDB
        return False

    # Шаг 4: Очистка промежуточного файла
    pdb_path.unlink(missing_ok=True)

    return True


# --- Оптимизированная основная функция ---


def main() -> None:
    """Основная функция для запуска параллельной подготовки лигандов."""
    src = config.GENERATED_MOLECULES_PATH
    if not src.exists():
        LOGGER.error(f"Файл с молекулами не найден: {src}")
        return

    # Создаем директорию для PDBQT файлов, если она не существует
    config.LIGAND_PDBQT_DIR.mkdir(exist_ok=True)

    df = pl.read_parquet(src)
    smiles_list = df["smiles"].to_list()
    total_ligands = len(smiles_list)
    LOGGER.info(f"Начинается подготовка {total_ligands} лигандов для докинга...")

    # Подготовка аргументов для параллельной обработки
    tasks = list(enumerate(smiles_list))

    # Создаем partial функцию, чтобы передать фиксированный аргумент `ligand_dir`
    worker_func = partial(_prepare_single, ligand_dir=config.LIGAND_PDBQT_DIR)

    # Запускаем пул процессов
    # Используем все доступные ядра процессора
    with Pool() as pool:
        # Используем imap для ленивой итерации и tqdm для прогресс-бара
        results = list(tqdm(pool.imap(worker_func, tasks), total=total_ligands, desc="Подготовка лигандов"))

    successful_count = sum(1 for r in results if r)
    failed_count = total_ligands - successful_count

    LOGGER.info("--- Статистика подготовки ---")
    LOGGER.info(f"Всего лигандов: {total_ligands}")
    LOGGER.info(f"Успешно обработано: {successful_count}")
    LOGGER.info(f"Не удалось обработать: {failed_count}")
    LOGGER.info(f"Готовые файлы находятся в директории: {config.LIGAND_PDBQT_DIR}")


if __name__ == "__main__":
    # Пример создания фейкового входного файла для тестирования
    if not config.GENERATED_MOLECULES_PATH.exists():
        LOGGER.info("Создание тестового файла generated_molecules.parquet...")
        test_smiles = [
            "CCO",  # Ethanol
            "c1ccccc1",  # Benzene
            "C#CCCN1C[C@@H]2C(NC(=O)c3ccc4c(c3)C(C)(C)CO4)[C@H]2C1",  # Проблемный SMILES
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # Caffeine
        ]
        test_df = pl.DataFrame({"smiles": test_smiles})
        test_df.write_parquet(config.GENERATED_MOLECULES_PATH)

    main()
