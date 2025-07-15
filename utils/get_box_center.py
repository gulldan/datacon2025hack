# file: get_box_center.py
import argparse
import sys
from pathlib import Path

import numpy as np
from logger import LOGGER

try:
    from Bio.PDB import PDBParser
except ImportError:
    LOGGER.error("Библиотека BioPython не найдена.")
    LOGGER.error("Пожалуйста, установите ее с помощью команды: pip install biopython")
    sys.exit(1)


def get_ligand_center(pdb_file: str | Path, ligand_name: str, verbose: bool = False) -> tuple[float, float, float] | None:
    """Вычисляет геометрический центр лиганда в PDB-файле (надежная версия).

    Args:
        pdb_file: Путь к PDB-файлу.
        ligand_name: Трехбуквенное имя лиганда (например, HMD).
        verbose: Флаг для вывода отладочной информации.

    Returns:
        Кортеж с координатами центра (x, y, z) или None, если лиганд не найден.
    """
    parser = PDBParser(QUIET=True)
    pdb_path = Path(pdb_file)
    if not pdb_path.exists():
        LOGGER.error(f"PDB-файл не найден по пути {pdb_path}")
        return None

    structure = parser.get_structure(pdb_path.stem, pdb_path)
    ligand_atoms = []
    found_ligands = set()

    for model in structure:
        for chain in model:
            for residue in chain:
                residue_id = residue.get_id()
                is_hetero = residue_id[0] != " " and residue_id[0] != "W"

                if verbose and is_hetero:
                    found_ligands.add(residue.get_resname())

                if is_hetero and residue.get_resname() == ligand_name:
                    ligand_atoms.extend(residue.get_unpacked_list())

    if verbose:
        LOGGER.info(f"Найденные гетеро-молекулы в файле: {sorted(found_ligands)}")

    if not ligand_atoms:
        LOGGER.error(f"Лиганд с именем '{ligand_name}' не найден в файле {pdb_path.name}")
        return None

    coords = [atom.get_coord() for atom in ligand_atoms]
    center = np.mean(coords, axis=0)

    return tuple(center)


def main() -> None:
    """Основная функция для запуска скрипта из командной строки."""
    parser = argparse.ArgumentParser(
        description="Вычисляет геометрический центр лиганда в PDB-файле",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--pdb_file", required=True, help="Путь к PDB-файлу (например, 3anq.pdb).")
    parser.add_argument("--ligand_name", required=True, help="Трехбуквенное имя лиганда (например, HMD).")
    parser.add_argument("--verbose", action="store_true", help="Показать все найденные лиганды в файле для отладки.")

    args = parser.parse_args()

    center_coords = get_ligand_center(args.pdb_file, args.ligand_name.upper(), args.verbose)

    if center_coords:
        x, y, z = center_coords
        LOGGER.info("--- Результаты ---")
        LOGGER.success(f"Лиганд '{args.ligand_name.upper()}' успешно найден.")
        LOGGER.info(f"Геометрический центр (X, Y, Z): ({x:.3f}, {y:.3f}, {z:.3f})")
        LOGGER.info("Скопируйте эти значения в ваш конфигурационный файл:")
        LOGGER.info(f"BOX_CENTER = ({x:.3f}, {y:.3f}, {z:.3f})")


if __name__ == "__main__":
    main()
