# step_04_hit_selection/run_hit_selection.py
import sys as _sys
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import (
    Chem,  # type: ignore
    )
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # type: ignore

# Ensure project root in PYTHONPATH when executed as script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in _sys.path:
    _sys.path.insert(0, str(ROOT_DIR))

import config
from utils.logger import LOGGER


# --- Имитация докинга ---
# На практике здесь будет вызов внешней программы (AutoDock Vina, smina, Glide)
# Мы сымитируем это, добавив случайный скор докинга.
def run_molecular_docking(smiles_list: list) -> dict:
    """Имитирует запуск молекулярного докинга.

    Args:
        smiles_list (list): Список SMILES для докинга.

    Returns:
        dict: Словарь {smiles: docking_score}.
    """
    LOGGER.info(f"Имитация молекулярного докинга для {len(smiles_list)} молекул...")
    # Более низкий скор докинга (более отрицательный) - лучшее связывание
    docking_scores = {smi: np.random.uniform(-10.0, -5.0) for smi in smiles_list}
    return docking_scores


def run_hit_selection_pipeline():
    """Основная функция для отбора итоговых молекул-хитов."""
    LOGGER.info("--- Запуск этапа 4: Отбор молекул-хитов ---")

    # 1. Загрузка сгенерированных молекул
    try:
        df = pl.read_parquet(config.GENERATED_MOLECULES_PATH)
    except FileNotFoundError:
        LOGGER.error(f"Файл {config.GENERATED_MOLECULES_PATH} не найден. Запустите этап 3.")
        return

    LOGGER.info(f"Загружено {len(df)} молекул для отбора.")

    # 2. Определение критериев отбора (фильтрация)
    # Эти пороги - ключевой элемент, который нужно обосновывать.
    # Они основаны на "правилах большого пальца" в медицинской химии.
    activity_threshold = 6.0  # pIC50 > 6.0 (IC50 ~1 µM)
    qed_threshold = 0.5       # Drug-likeness > 0.5
    sa_score_threshold = 4.0  # Синтезируемость < 4.0 (чем ниже, тем проще)
    bbbp_threshold = 0.7      # Вероятность прохождения ГЭБ > 70%

    # Применяем фильтры с параметрами из конфигурации для DYRK1A
    activity_min = config.HIT_SELECTION_FILTERS["activity_filters"]["predicted_pic50"]["min"]
    qed_min = config.HIT_SELECTION_FILTERS["drug_likeness_filters"]["qed"]["min"]
    sa_max = config.HIT_SELECTION_FILTERS["synthetic_accessibility"]["sa_score"]["max"]
    bbbp_min = config.HIT_SELECTION_FILTERS["bbb_permeability"]["min"]

    LOGGER.info("Применение фильтров для отбора хитов DYRK1A...")
    LOGGER.info(f"Критерии для DYRK1A (болезнь Альцгеймера): pIC50 > {activity_min}, QED > {qed_min}, SA_score < {sa_max}, BBBP > {bbbp_min}")

    hits = df.filter(
        (pl.col("predicted_pIC50") > activity_min) &
        (pl.col("qed") > qed_min) &
        (pl.col("sa_score") < sa_max) &
        (pl.col("bbbp_prob") > bbbp_min)
    )

    LOGGER.info(f"Найдено {len(hits)} молекул после первичной фильтрации.")

    if len(hits) == 0:
        LOGGER.warning("Не найдено молекул, удовлетворяющих критериям. Попробуйте ослабить фильтры.")
        return

    # 3. Молекулярный докинг (для отфильтрованных кандидатов)
    if config.VINA_RESULTS_PATH.exists():
        LOGGER.info(f"Using real AutoDock Vina scores from {config.VINA_RESULTS_PATH}")
        docking_df = pl.read_parquet(config.VINA_RESULTS_PATH)
        docking_df = docking_df.rename({"ligand_id": "smiles"}) if "ligand_id" in docking_df.columns else docking_df
    else:
        LOGGER.warning("Vina scores not found – running complete docking pipeline...")

        # Запускаем полный pipeline подготовки и докинга
        try:
            # 1. Подготовка белка
            LOGGER.info("Preparing protein receptor...")
            from step_04_hit_selection.protein_prep import main as prepare_protein
            prepare_protein()

            # 2. Подготовка лигандов (только отфильтрованных)
            LOGGER.info(f"Preparing {len(hits)} filtered ligands for docking...")
            from step_04_hit_selection.ligand_prep import is_valid_pdbqt, pdb_to_pdbqt, smiles_to_3d_pdb

            # Создаем временный файл с отфильтрованными молекулами
            filtered_molecules_path = config.GENERATED_MOLECULES_PATH.parent / "filtered_molecules.parquet"
            hits.write_parquet(filtered_molecules_path)

            # Подготавливаем только отфильтрованные молекулы
            ligand_mapping = {}  # Маппинг SMILES -> ligand_id
            for idx, smi in enumerate(hits["smiles"]):
                pdb_path = config.LIGAND_PDBQT_DIR / f"filtered_lig_{idx}.pdb"
                pdbqt_path = pdb_path.with_suffix(".pdbqt")

                if pdbqt_path.exists():
                    ligand_mapping[smi] = f"filtered_lig_{idx}"
                    continue

                if not smiles_to_3d_pdb(smi, pdb_path):
                    LOGGER.warning(f"Failed to generate 3D for {smi}")
                    continue

                pdb_to_pdbqt(pdb_path, pdbqt_path)

                if not is_valid_pdbqt(pdbqt_path):
                    LOGGER.warning(f"OpenBabel failed to create valid PDBQT for {smi}")
                    pdbqt_path.unlink(missing_ok=True)
                    pdb_path.unlink(missing_ok=True)
                    continue

                ligand_mapping[smi] = f"filtered_lig_{idx}"

            LOGGER.info(f"Successfully prepared {len(ligand_mapping)} ligands for docking")

            # 3. Запуск докинга для отфильтрованных лигандов
            LOGGER.info(f"Running AutoDock Vina for {len(ligand_mapping)} ligands...")
            import shutil

            from step_04_hit_selection.run_vina import dock_ligand, has_atoms

            # Проверяем наличие Vina
            if not shutil.which("vina"):
                raise FileNotFoundError("AutoDock Vina not found in PATH")

            # Докинг отфильтрованных лигандов
            docking_results = []
            for smi, ligand_id in ligand_mapping.items():
                lig_pdbqt = config.LIGAND_PDBQT_DIR / f"{ligand_id}.pdbqt"
                if not lig_pdbqt.exists() or not has_atoms(lig_pdbqt):
                    continue

                out_pdbqt = lig_pdbqt.with_name(lig_pdbqt.stem + "_dock.pdbqt")
                log_path = lig_pdbqt.with_suffix(".log")

                score = dock_ligand(lig_pdbqt, out_pdbqt, log_path)
                if score is not None:
                    docking_results.append((smi, score))

            # Сохраняем результаты
            if docking_results:
                docking_df = pl.DataFrame(docking_results, schema=["smiles", "docking_score"])
                docking_df.write_parquet(config.VINA_RESULTS_PATH)
                LOGGER.info(f"Docking completed for {len(docking_results)} ligands")
            else:
                raise RuntimeError("No successful docking results")

            # 4. Загрузка результатов (уже создан выше)
            LOGGER.info("Successfully completed docking pipeline")

        except Exception as e:
            LOGGER.error(f"Docking pipeline failed: {e}")
            LOGGER.warning("Falling back to random docking stub.")
            smiles_to_dock = hits["smiles"].to_list()
            docking_results = run_molecular_docking(smiles_to_dock)
            docking_df = pl.DataFrame({
                "smiles": list(docking_results.keys()),
                "docking_score": list(docking_results.values())
            })

    hits = hits.join(docking_df, on="smiles")

    # Фильтруем по скору докинга с параметрами из конфигурации
    docking_threshold = config.HIT_SELECTION_FILTERS["docking_filters"]["binding_energy"]["max"]
    LOGGER.info(f"Применение фильтра по скору докинга для DYRK1A: < {docking_threshold} kcal/mol")
    final_hits = hits.filter(pl.col("docking_score") < docking_threshold)
    LOGGER.info(f"Осталось {len(final_hits)} молекул после фильтрации по докингу.")

    if len(final_hits) == 0:
        LOGGER.warning("Не найдено молекул после докинга. Попробуйте ослабить порог.")
        return

    # 4. Обеспечение разнообразия (Diversity Picking) с параметрами из конфигурации
    # Чтобы финальный список не состоял из очень похожих молекул.
    max_hits = config.PIPELINE_PARAMETERS["hit_selection"]["max_hits"]
    num_final_hits = min(max_hits, len(final_hits))
    LOGGER.info(f"Отбор {num_final_hits} наиболее разнообразных молекул DYRK1A из кандидатов...")

    # Простая альтернатива MaxMinPicker для diversity picking
    import random

    from rdkit.Chem import DataStructs

    mols = [Chem.MolFromSmiles(s) for s in final_hits["smiles"]]  # type: ignore[attr-defined]
    gen = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    fps = []
    for m in mols:
        if m is not None:
            bv = gen.GetFingerprint(m)
        else:
            bv = None
        fps.append(bv)

    # Быстрый MaxMin diversity picking для больших наборов данных
    def fast_maxmin_pick(fingerprints, num_picks, seed=42):
        random.seed(seed)
        valid_indices = [i for i, fp in enumerate(fingerprints) if fp is not None]

        if len(valid_indices) <= num_picks:
            return valid_indices

        # Если слишком много молекул, сначала делаем случайную выборку
        if len(valid_indices) > 1000:
            LOGGER.info(f"Large dataset ({len(valid_indices)} molecules), using random pre-selection...")
            valid_indices = random.sample(valid_indices, min(1000, len(valid_indices)))

        # Начинаем с случайной молекулы
        selected = [random.choice(valid_indices)]

        for _ in range(min(num_picks - 1, len(valid_indices) - 1)):
            max_min_dist = -1
            best_idx = -1

            for i in valid_indices:
                if i in selected:
                    continue

                # Найти минимальное расстояние до уже выбранных
                min_dist = float("inf")
                for j in selected:
                    similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                    distance = 1 - similarity
                    min_dist = min(min_dist, distance)

                # Выбрать молекулу с максимальным минимальным расстоянием
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i

            if best_idx != -1:
                selected.append(best_idx)

        return selected

    pick_indices = fast_maxmin_pick(fps, num_final_hits)

    diverse_hits_df = final_hits[pick_indices]

    # 5. Сохранение итогового списка
    final_df = diverse_hits_df.sort("docking_score", descending=False)
    final_df.write_parquet(config.FINAL_HITS_PATH)

    LOGGER.info(f"Итоговый список из {len(final_df)} молекул-хитов сохранен в {config.FINAL_HITS_PATH}")
    LOGGER.info("Финальные кандидаты:")
    LOGGER.info(f"\n{final_df}")
    LOGGER.info("--- Этап 4 завершен ---")


if __name__ == "__main__":
    run_hit_selection_pipeline()
