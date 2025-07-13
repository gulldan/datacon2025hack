# step_04_hit_selection/run_hit_selection.py
import polars as pl
from rdkit import Chem
from rdkit.SimDivFilters import MaxMinPicker

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
    activity_threshold = 6.5  # pIC50 > 6.5 (соответствует IC50 < ~316 нМ)
    qed_threshold = 0.5       # Drug-likeness > 0.5
    sa_score_threshold = 4.0  # Синтезируемость < 4.0 (чем ниже, тем проще)
    bbbp_threshold = 0.7      # Вероятность прохождения ГЭБ > 70%

    LOGGER.info("Применение фильтров для отбора хитов...")
    LOGGER.info(f"Критерии: pIC50 > {activity_threshold}, QED > {qed_threshold}, SA_score < {sa_score_threshold}, BBBP > {bbbp_threshold}")

    hits_df = df.filter(
        (pl.col("predicted_pIC50") > activity_threshold) &
        (pl.col("qed") > qed_threshold) &
        (pl.col("sa_score") < sa_score_threshold) &
        (pl.col("bbbp_prob") > bbbp_threshold)
    )
    LOGGER.info(f"Найдено {len(hits_df)} молекул после первичной фильтрации.")

    if len(hits_df) == 0:
        LOGGER.warning("Не найдено молекул, удовлетворяющих критериям. Попробуйте ослабить фильтры.")
        return

    # 3. Молекулярный докинг (для отфильтрованных кандидатов)
    smiles_to_dock = hits_df["smiles"].to_list()
    docking_results = run_molecular_docking(smiles_to_dock)

    docking_df = pl.DataFrame({
        "smiles": list(docking_results.keys()),
        "docking_score": list(docking_results.values())
    })

    hits_df = hits_df.join(docking_df, on="smiles")

    # Фильтруем по скору докинга
    docking_threshold = -7.5 # ккал/моль
    LOGGER.info(f"Применение фильтра по скору докинга: < {docking_threshold} kcal/mol")
    final_hits = hits_df.filter(pl.col("docking_score") < docking_threshold)
    LOGGER.info(f"Осталось {len(final_hits)} молекул после фильтрации по докингу.")

    if len(final_hits) == 0:
        LOGGER.warning("Не найдено молекул после докинга. Попробуйте ослабить порог.")
        return

    # 4. Обеспечение разнообразия (Diversity Picking)
    # Чтобы финальный список не состоял из очень похожих молекул.
    num_final_hits = min(10, len(final_hits)) # Выберем до 10 самых разнообразных
    LOGGER.info(f"Отбор {num_final_hits} наиболее разнообразных молекул из кандидатов...")

    mols = [Chem.MolFromSmiles(s) for s in final_hits["smiles"]]
    fps = [GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]

    picker = MaxMinPicker()
    pick_indices = list(picker.LazyPick(fps, len(fps), num_final_hits))

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
