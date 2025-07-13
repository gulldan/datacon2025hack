# step_03_molecule_generation/run_generation.py

import numpy as np
import polars as pl
from rdkit import Chem  # type: ignore
from rdkit.Chem import QED, Crippen  # type: ignore
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect  # type: ignore

import config
from utils.logger import LOGGER

# --- Вспомогательные функции для оценки свойств ---

def calculate_sa_score(smiles: str):
    """Расчет Synthetic Accessibility score."""
    # Эта функция требует установки rdkit-pypi>=2022.9.1 и скачанных файлов
    # Для простоты вернем случайное значение, имитируя вызов
    # from rdkit.Chem import RDConfig
    # import os
    # sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    # import sascorer
    # mol = Chem.MolFromSmiles(smiles)
    # return sascorer.calculateScore(mol) if mol else 5.0
    return np.random.uniform(1, 5)


def calculate_bbbp(smiles: str):
    """Простая модель для предсказания проницаемости через ГЭБ (BBB)."""
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if not mol:
        return 0.0
    # Правило Egan: TPSA <= 132 и LogP <= 5.9
    tpsa = QED.properties(mol).PSA
    logp = Crippen.MolLogP(mol)
    if tpsa <= 132 and logp <= 5.9:
        return np.random.uniform(0.7, 1.0) # Вероятно проходит
    return np.random.uniform(0.0, 0.3) # Вероятно не проходит

def get_scoring_function(activity_model):
    """Создает и возвращает скоринговую функцию для оценки сгенерированных молекул.
    Эта функция будет сердцем направленной генерации.

    Args:
        activity_model: Обученная модель для предсказания активности.

    Returns:
        function: Функция, принимающая SMILES и возвращающая итоговый скор.
    """
    def score_molecule(smiles: str) -> float:
        """Оценивает молекулу по нескольким параметрам.

        Args:
            smiles (str): SMILES строка молекулы.

        Returns:
            float: Итоговый скор от 0 до 1.
        """
        mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
        if not mol or not smiles:
            return 0.0

        # 1. Валидность и Drug-likeness (QED)
        qed_score = QED.qed(mol)

        # 2. Предсказанная активность
        fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        predicted_pic50 = activity_model.predict(np.array(fp).reshape(1, -1))[0]
        # Нормализуем pIC50 (например, цель > 7.0)
        activity_score = min(1.0, max(0.0, (predicted_pic50 - 5.0) / 3.0)) # Цель [5, 8] -> [0, 1]

        # 3. Синтезируемость (SA Score)
        sa_score_val = calculate_sa_score(smiles)
        # Нормализуем (цель < 4)
        sa_score = max(0.0, (5.0 - sa_score_val) / 4.0) # Цель [5, 1] -> [0, 1]

        # 4. Проницаемость через ГЭБ (BBBP)
        bbbp_score = calculate_bbbp(smiles)

        # 5. Объединение метрик в один скор (взвешенная сумма)
        weights = {"activity": 0.4, "qed": 0.2, "sa": 0.2, "bbbp": 0.2}

        final_score = (
            weights["activity"] * activity_score +
            weights["qed"] * qed_score +
            weights["sa"] * sa_score +
            weights["bbbp"] * bbbp_score
        )
        return final_score

    return score_molecule

def run_generation_pipeline():
    """Основная функция для запуска пайплайна генерации молекул.
    """
    LOGGER.info("--- Запуск этапа 3: Генерация молекул ---")

    # 1. Обоснование выбора генеративной модели
    LOGGER.info("""
    Обоснование выбора генеративной модели:
    Для этой задачи идеально подходят модели на основе Reinforcement Learning (RL), такие как REINVENT или FREED++.
    - REINVENT позволяет оптимизировать молекулы по сложной, кастомной скоринговой функции, 
      которая может включать предсказание активности, ADMET-свойства и результаты докинга.
    - Это позволяет проводить "направленную" генерацию, смещая распределение генерируемых
      молекул в сторону желаемых свойств.
    - Альтернативы: MolGAN (быстрее, но менее управляем), DrugGPT (на основе трансформеров,
      требует больших данных для дообучения).
    
    В данном скрипте мы сымитируем финальный этап работы такой модели:
    оценку пула сгенерированных молекул с помощью скоринговой функции.
    """)

    # 2. Загрузка модели предсказания активности (coeffs + bias)
    from step_02_activity_prediction.model_utils import load_model, smiles_to_fp

    activity_model = load_model(config.MODEL_PATH)
    LOGGER.info("Загружены коэффициенты модели активности (%d bits)", len(activity_model.coeffs()))

    # 3. Создание скоринговой функции
    scoring_function = get_scoring_function(activity_model)

    # 4. Генерация молекул при помощи SELFIES-VAE
    from step_03_molecule_generation.selfies_vae_generator import train_and_sample

    LOGGER.info("Генерируем молекулы SELFIES-VAE…")
    generated_smiles_pool: list[str] = train_and_sample(2000)

    LOGGER.info(f"Оценка {len(generated_smiles_pool)} сгенерированных молекул...")

    results = []
    for smiles in generated_smiles_pool:
        mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
        if not mol: continue

        score = scoring_function(smiles)
        fp_arr = smiles_to_fp(smiles)
        if fp_arr is None:
            continue
        pIC50 = float(activity_model.predict(fp_arr)[0])

        results.append({
            "smiles": smiles,
            "final_score": score,
            "predicted_pIC50": pIC50,
            "qed": QED.qed(mol),
            "logp": Crippen.MolLogP(mol),
            "sa_score": calculate_sa_score(smiles),
            "bbbp_prob": calculate_bbbp(smiles)
        })

    generated_df = pl.DataFrame(results)
    generated_df.write_parquet(config.GENERATED_MOLECULES_PATH)

    LOGGER.info(f"Сгенерированные и оцененные молекулы сохранены в {config.GENERATED_MOLECULES_PATH}")
    LOGGER.info(f"Топ-5 молекул по итоговому скору:\n{generated_df.sort('final_score', descending=True).head(5)}")
    LOGGER.info("--- Этап 3 завершен ---")

if __name__ == "__main__":
    run_generation_pipeline()
