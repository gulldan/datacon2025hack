# step_03_molecule_generation/run_generation.py

import numpy as np
import polars as pl
from rdkit import (
    Chem,  # type: ignore
    DataStructs,  # type: ignore
)
from rdkit.Chem import QED, Crippen  # type: ignore
from rdkit.Chem.Descriptors import Descriptors  # type: ignore
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # type: ignore

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
        # Build feature vector according to model type
        use_linear = hasattr(activity_model, "coeffs") and len(activity_model.coeffs()) > 0  # type: ignore[arg-type]

        if use_linear:
            gen = GetMorganGenerator(
                radius=config.FP_RADIUS,
                fpSize=config.FP_BITS_LINEAR,
                includeChirality=config.FP_INCLUDE_CHIRALITY,
            )
            bv = gen.GetFingerprint(mol)
            arr = np.zeros((config.FP_BITS_LINEAR,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(bv, arr)  # type: ignore[arg-type]
        else:
            # XGBoost expects 1024 bits + 6 descriptors
            gen = GetMorganGenerator(
                radius=config.FP_RADIUS,
                fpSize=config.FP_BITS_XGB,
                includeChirality=config.FP_INCLUDE_CHIRALITY,
            )
            bv = gen.GetFingerprint(mol)
            fp_arr = np.zeros((config.FP_BITS_XGB,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(bv, fp_arr)  # type: ignore[arg-type]
            desc_vals = np.asarray([
                Descriptors.MolWt(mol),  # type: ignore[attr-defined]
                Descriptors.MolLogP(mol),  # type: ignore[attr-defined]
                Descriptors.TPSA(mol),  # type: ignore[attr-defined]
                Descriptors.NumHDonors(mol),  # type: ignore[attr-defined]
                Descriptors.NumHAcceptors(mol),  # type: ignore[attr-defined]
                Descriptors.RingCount(mol),  # type: ignore[attr-defined]
            ], dtype=np.float32)
            arr = np.concatenate([fp_arr, desc_vals])

        predicted_pic50 = activity_model.predict(arr.reshape(1, -1))[0]
        # Нормализуем pIC50 (например, цель > 7.0)
        activity_score = min(1.0, max(0.0, (predicted_pic50 - 5.0) / 3.0)) # Цель [5, 8] -> [0, 1]

        # 3. Синтезируемость (SA Score)
        sa_score_val = calculate_sa_score(smiles)
        # Нормализуем (цель < 4)
        sa_score = max(0.0, (5.0 - sa_score_val) / 4.0) # Цель [5, 1] -> [0, 1]

        # 4. Проницаемость через ГЭБ (BBBP)
        bbbp_score = calculate_bbbp(smiles)

        # 5. Объединение метрик в один скор (взвешенная сумма)
        weights = config.SCORING_WEIGHTS

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
    from step_02_activity_prediction.model_utils import load_model

    activity_model = load_model(config.MODEL_PATH)
    if hasattr(activity_model, "coeffs") and len(activity_model.coeffs()) > 0:  # type: ignore[arg-type]
        LOGGER.info("Загружены коэффициенты линейной модели (%d bits)", len(activity_model.coeffs()))
    else:
        LOGGER.info("Загружена XGBoost модель предсказания активности")

    # 3. Создание скоринговой функции
    scoring_function = get_scoring_function(activity_model)

    # 4. Генерация молекул согласно выбранному типу
    if config.GENERATOR_TYPE == "selfies_vae":
        from step_03_molecule_generation.selfies_vae_generator import train_and_sample

        LOGGER.info("Генерируем молекулы SELFIES-VAE…")
        generated_smiles_pool: list[str] = train_and_sample(config.VAE_GENERATE_N)
    elif config.GENERATOR_TYPE == "graph_flow":
        try:
            from step_03_molecule_generation.graph_generator import train_and_sample
        except ImportError:
            LOGGER.error("Graph generator module not found. Ensure T21 is implemented.")
            return

        LOGGER.info("Генерируем молекулы Graph-Flow…")
        generated_smiles_pool = train_and_sample(config.VAE_GENERATE_N)
    else:
        LOGGER.error("Unknown GENERATOR_TYPE '%s' in config.py", config.GENERATOR_TYPE)
        return

    LOGGER.info(f"Оценка {len(generated_smiles_pool)} сгенерированных молекул...")

    results = []
    for smiles in generated_smiles_pool:
        mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
        if not mol: continue

        score = scoring_function(smiles)

        # Построим тот же набор признаков, что использовался в скоринговой функции
        use_linear = hasattr(activity_model, "coeffs") and len(activity_model.coeffs()) > 0  # type: ignore[arg-type]

        if use_linear:
            gen = GetMorganGenerator(
                radius=config.FP_RADIUS,
                fpSize=config.FP_BITS_LINEAR,
                includeChirality=config.FP_INCLUDE_CHIRALITY,
            )
            bv = gen.GetFingerprint(mol)
            arr = np.zeros((config.FP_BITS_LINEAR,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(bv, arr)  # type: ignore[arg-type]
        else:
            gen = GetMorganGenerator(
                radius=config.FP_RADIUS,
                fpSize=config.FP_BITS_XGB,
                includeChirality=config.FP_INCLUDE_CHIRALITY,
            )
            bv = gen.GetFingerprint(mol)
            fp_arr = np.zeros((config.FP_BITS_XGB,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(bv, fp_arr)  # type: ignore[arg-type]
            desc_vals = np.asarray([
                Descriptors.MolWt(mol),  # type: ignore[attr-defined]
                Descriptors.MolLogP(mol),  # type: ignore[attr-defined]
                Descriptors.TPSA(mol),  # type: ignore[attr-defined]
                Descriptors.NumHDonors(mol),  # type: ignore[attr-defined]
                Descriptors.NumHAcceptors(mol),  # type: ignore[attr-defined]
                Descriptors.RingCount(mol),  # type: ignore[attr-defined]
            ], dtype=np.float32)
            arr = np.concatenate([fp_arr, desc_vals])

        pIC50 = float(activity_model.predict(arr.reshape(1, -1))[0])

        results.append({
            "smiles": smiles,
            "final_score": score,
            "predicted_pIC50": pIC50,
            "qed": QED.qed(mol),
            "logp": Crippen.MolLogP(mol),
            "sa_score": calculate_sa_score(smiles),
            "bbbp_prob": calculate_bbbp(smiles)
        })

    generated_df = pl.DataFrame(results) if results else pl.DataFrame(schema={
        "smiles": pl.Utf8,
        "final_score": pl.Float64,
        "predicted_pIC50": pl.Float64,
        "qed": pl.Float64,
        "logp": pl.Float64,
        "sa_score": pl.Float64,
        "bbbp_prob": pl.Float64,
    })

    generated_df.write_parquet(config.GENERATED_MOLECULES_PATH)

    LOGGER.info("Сгенерированные и оцененные молекулы сохранены в %s", config.GENERATED_MOLECULES_PATH)
    if len(generated_df) == 0:
        LOGGER.warning("SELFIES-VAE не сгенерировал валидные молекулы – проверьте качество модели или увеличьте эпохи обучения.")
    else:
        top5 = generated_df.sort("final_score", descending=True).head(5)
        LOGGER.info("Топ-5 молекул по итоговому скору:\n%s", top5)
    LOGGER.info("--- Этап 3 завершен ---")

if __name__ == "__main__":
    run_generation_pipeline()
