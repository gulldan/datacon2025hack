# file: prepare_ligands.py
"""Подготовка лигандов для докинга.

Читает отфильтрованные молекулы из Parquet файла, генерирует для каждой 3D-структуру
с помощью RDKit, а затем конвертирует в формат PDBQT с помощью OpenBabel.
Работает в многопроцессном режиме для максимальной скорости.
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# Добавляем корневую директорию в путь
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from utils.logger import LOGGER


def prepare_single_ligand(args: tuple[int, str, Path]) -> tuple[str, bool]:
    """Выполняет полный цикл подготовки для одного лиганда: SMILES -> PDBQT.
    Возвращает кортеж (ID молекулы, статус успеха).
    """
    idx, smiles, output_dir = args
    mol_id = f"mol_{idx}"
    output_pdbqt_path = output_dir / f"{mol_id}.pdbqt"

    # Пропускаем, если файл уже существует
    if output_pdbqt_path.exists():
        return mol_id, True

    # --- Этап 1: Генерация 3D-структуры (PDB) ---
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        LOGGER.warning(f"[{mol_id}] RDKit не смог прочитать SMILES: {smiles}")
        return mol_id, False

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D  # Для воспроизводимости
    if AllChem.EmbedMolecule(mol, params) == -1:
        LOGGER.warning(f"[{mol_id}] RDKit не смог сгенерировать 3D-конформер для {smiles}")
        return mol_id, False

    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as e:
        LOGGER.warning(f"[{mol_id}] Ошибка оптимизации 3D для {smiles}: {e}")
        return mol_id, False

    # Используем временный PDB файл
    temp_pdb_path = output_pdbqt_path.with_suffix(".tmp.pdb")
    Chem.MolToPDBFile(mol, str(temp_pdb_path))

    # --- Этап 2: Конвертация PDB в PDBQT с помощью OpenBabel ---
    cmd = ["obabel", str(temp_pdb_path), "-O", str(output_pdbqt_path), "--partialcharge", "gasteiger"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        error_msg = e.stderr.strip() if hasattr(e, "stderr") else str(e)
        LOGGER.error(f"[{mol_id}] OpenBabel не смог конвертировать {smiles}. Ошибка: {error_msg}")
        temp_pdb_path.unlink(missing_ok=True)
        return mol_id, False
    finally:
        # Гарантированно удаляем временный файл
        temp_pdb_path.unlink(missing_ok=True)

    # --- Этап 3: Валидация ---
    if not output_pdbqt_path.exists() or output_pdbqt_path.stat().st_size == 0:
        LOGGER.warning(f"[{mol_id}] OpenBabel создал пустой PDBQT файл для {smiles}.")
        return mol_id, False

    return mol_id, True


def main():
    """Основная функция для запуска параллельной подготовки лигандов."""
    # Убедимся, что директория для лигандов существует
    config.LIGAND_PDBQT_DIR.mkdir(exist_ok=True)

    # Загружаем отфильтрованные молекулы
    source_file = config.GENERATED_MOLECULES_PATH  # Убедитесь, что путь в config.py правильный
    if not source_file.exists():
        LOGGER.error(f"Файл с отфильтрованными молекулами не найден: {source_file}")
        return

    df = pl.read_parquet(source_file)
    # Предполагаем, что колонка со SMILES называется 'smiles'
    smiles_list = df["smiles"].to_list()

    LOGGER.info(f"Начинается подготовка {len(smiles_list)} лигандов...")

    # Подготовка аргументов для параллельной обработки
    tasks = [(i, smi, config.LIGAND_PDBQT_DIR) for i, smi in enumerate(smiles_list)]

    successful_count = 0
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(prepare_single_ligand, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Подготовка лигандов"):
            try:
                _, success = future.result()
                if success:
                    successful_count += 1
            except Exception as e:
                LOGGER.error(f"Критическая ошибка в дочернем процессе: {e}")

    LOGGER.info("--- Статистика подготовки лигандов ---")
    LOGGER.info(f"Всего молекул для обработки: {len(smiles_list)}")
    LOGGER.info(f"Успешно подготовлено: {successful_count}")
    LOGGER.info(f"Не удалось подготовить: {len(smiles_list) - successful_count}")
    LOGGER.info(f"Готовые PDBQT файлы находятся в: {config.LIGAND_PDBQT_DIR}")


if __name__ == "__main__":
    main()
