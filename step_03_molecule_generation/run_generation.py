# step_03_molecule_generation/run_generation.py

# Предполагается, что у вас есть логгер
import logging
from collections.abc import Callable

import numpy as np
import polars as pl
from rdkit import (
    Chem,  # type: ignore
    DataStructs,  # type: ignore
)
from rdkit.Chem import (  # type: ignore
    QED,
    Crippen,
    Descriptors,  # type: ignore
)
from rdkit.Chem import rdMolDescriptors as rdMD  # ← новые версии
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # type: ignore

import config
from utils.logger import LOGGER

LOGGER = logging.getLogger(__name__)


def _safe(rdkit_fun: str, fallback_fun: Callable[[Chem.Mol], int]) -> Callable[[Chem.Mol], int]:
    """Return the first available descriptor function."""
    return getattr(Descriptors, rdkit_fun, fallback_fun)


_calc_rot_bonds = _safe("NumRotatableBonds", rdMD.CalcNumRotatableBonds)
_calc_spiro_atoms = _safe("NumSpiroAtoms", rdMD.CalcNumSpiroAtoms)
_calc_bridgehead = _safe("NumBridgeheadAtoms", rdMD.CalcNumBridgeheadAtoms)


def _calculate_sa_score(mol: Chem.Mol) -> float:
    """Approximate Synthetic Accessibility (SA) score on a 1‒10 scale.
    ➜ Бóльшие значения — сложнее/дороже синтез.
    """
    try:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass

        score = 1.0

        # 1) размер
        num_atoms = mol.GetNumAtoms()
        if num_atoms > 50:
            score += 2.0
        elif num_atoms > 30:
            score += 1.5
        elif num_atoms > 20:
            score += 1.0

        # 2) циклы
        ring_info = mol.GetRingInfo()
        ring_count = ring_info.NumRings()
        score += 0.5 * ring_count
        complex_rings = sum(1 for ring in ring_info.AtomRings() if len(ring) > 6)
        score += 0.8 * complex_rings

        # 3) гибкость + «странные» атомы
        score += 0.3 * _calc_rot_bonds(mol)
        score += 1.0 * _calc_spiro_atoms(mol)
        score += 0.8 * _calc_bridgehead(mol)

        # ИСПРАВЛЕНИЕ ЗДЕСЬ:
        stereo_centers = Chem.FindPotentialStereo(mol, flagPossible=True)
        score += 0.4 * len(stereo_centers)

        heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() not in {"C", "H"})
        score += 0.2 * heteroatoms

        # 4) «хитрые» функциональные группы
        complex_smarts = [
            "C(=O)N",
            "C(=O)O",
            "C(=O)[Cl,Br,I]",
            "C#N",
            "C#C",
            "C=C=C",
        ]
        complex_groups = sum(mol.HasSubstructMatch(Chem.MolFromSmarts(p)) for p in complex_smarts)
        score += 0.3 * complex_groups

        return float(max(1.0, min(10.0, score)))

    except Exception as exc:
        LOGGER.warning("SA-score fallback (returned 5.0) — error: %s", exc)
        return 5.0


def calculate_sa_score(smiles: str):
    """Расчет Synthetic Accessibility score."""
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if not mol:
        LOGGER.warning(f"Failed to calculate SA score, returning 5.0: {smiles}")
        return 5.0  # Fallback to heuristic approximation
    # Fallback to heuristic approximation
    return _calculate_sa_score(mol)


def calculate_bbbp(smiles: str):
    """Детерминированная модель для предсказания проницаемости через ГЭБ (BBB).

    Основана на научно обоснованных правилах:
    - Lipinski Rule of Five (модифицированная для ЦНС)
    - Egan Rule (TPSA <= 132, LogP <= 5.9)
    - Veber Rule (RotBonds <= 10, TPSA <= 140)
    - Молекулярная масса < 500 для ЦНС препаратов
    """
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if not mol:
        return 0.0

    # Получаем молекулярные дескрипторы
    molwt = Descriptors.MolWt(mol)  # type: ignore[attr-defined]
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)  # type: ignore[attr-defined]
    hbd = Descriptors.NumHDonors(mol)  # type: ignore[attr-defined]
    hba = Descriptors.NumHAcceptors(mol)  # type: ignore[attr-defined]
    rotbonds = Descriptors.NumRotatableBonds(mol)  # type: ignore[attr-defined]

    # Базовый скор (0-1)
    score = 0.0

    # 1. Правило Egan (основное для ЦНС)
    if tpsa <= 132 and logp <= 5.9:
        score += 0.4
    elif tpsa <= 140 and logp <= 5.0:
        score += 0.3
    elif tpsa <= 150 and logp <= 4.0:
        score += 0.2

    # 2. Правило Veber
    if rotbonds <= 10 and tpsa <= 140:
        score += 0.2

    # 3. Молекулярная масса (оптимально для ЦНС: 150-500)
    if 150 <= molwt <= 500:
        score += 0.2
    elif 100 <= molwt <= 600:
        score += 0.1

    # 4. Водородные связи (оптимально для ЦНС)
    if hbd <= 3 and hba <= 7:
        score += 0.1
    elif hbd <= 5 and hba <= 10:
        score += 0.05

    # 5. LogP оптимизация для ЦНС (1-4 оптимально)
    if 1.0 <= logp <= 4.0:
        score += 0.1
    elif 0.5 <= logp <= 5.0:
        score += 0.05

    # Нормализуем и ограничиваем результат
    return min(1.0, max(0.0, score))


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
            desc_vals = np.asarray(
                [
                    Descriptors.MolWt(mol),  # type: ignore[attr-defined]
                    Descriptors.MolLogP(mol),  # type: ignore[attr-defined]
                    Descriptors.TPSA(mol),  # type: ignore[attr-defined]
                    Descriptors.NumHDonors(mol),  # type: ignore[attr-defined]
                    Descriptors.NumHAcceptors(mol),  # type: ignore[attr-defined]
                    Descriptors.RingCount(mol),  # type: ignore[attr-defined]
                ],
                dtype=np.float32,
            )
            arr = np.concatenate([fp_arr, desc_vals])

        predicted_pic50 = activity_model.predict(arr.reshape(1, -1))[0]
        # Нормализуем pIC50 (например, цель > 7.0)
        activity_score = min(1.0, max(0.0, (predicted_pic50 - 5.0) / 3.0))  # Цель [5, 8] -> [0, 1]

        # 3. Синтезируемость (SA Score)
        sa_score_val = calculate_sa_score(smiles)
        # Нормализуем (цель < 4)
        sa_score = max(0.0, (5.0 - sa_score_val) / 4.0)  # Цель [5, 1] -> [0, 1]

        # 4. Проницаемость через ГЭБ (BBBP)
        bbbp_score = calculate_bbbp(smiles)

        # 5. Объединение метрик в один скор (взвешенная сумма для DYRK1A)
        weights = config.SCORING_WEIGHTS

        final_score = (
            weights["activity"] * activity_score
            + weights["qed"] * qed_score
            + weights["sa"] * sa_score
            + weights["bbbp"] * bbbp_score
        )

        # Добавляем компонент селективности если доступен
        if "selectivity" in weights:
            # Простая эвристика селективности для DYRK1A (можно улучшить)
            selectivity_score = min(1.0, activity_score * 0.8)  # Предполагаем корреляцию
            final_score += weights["selectivity"] * selectivity_score

        return final_score

    return score_molecule


def run_generation_pipeline():
    """Основная функция для запуска пайплайна генерации молекул."""
    LOGGER.info("--- Запуск этапа 3: Генерация молекул ---")

    # 1. Обоснование выбора генеративной модели
    # LOGGER.info("""
    # Обоснование выбора генеративной модели:
    # Для этой задачи идеально подходят модели на основе Reinforcement Learning (RL), такие как REINVENT или FREED++.
    # - REINVENT позволяет оптимизировать молекулы по сложной, кастомной скоринговой функции,
    #   которая может включать предсказание активности, ADMET-свойства и результаты докинга.
    # - Это позволяет проводить "направленную" генерацию, смещая распределение генерируемых
    #   молекул в сторону желаемых свойств.
    # - Альтернативы: MolGAN (быстрее, но менее управляем), DrugGPT (на основе трансформеров,
    #   требует больших данных для дообучения).

    # В данном скрипте мы сымитируем финальный этап работы такой модели:
    # оценку пула сгенерированных молекул с помощью скоринговой функции.
    # """)

    # 2. Загрузка модели предсказания активности (coeffs + bias)
    from step_02_activity_prediction.model_utils import load_model

    activity_model = load_model(config.MODEL_PATH)
    if hasattr(activity_model, "coeffs") and len(activity_model.coeffs()) > 0:  # type: ignore[arg-type]
        LOGGER.info("Загружены коэффициенты линейной модели (%d bits)", len(activity_model.coeffs()))
    else:
        LOGGER.info("Загружена XGBoost модель предсказания активности")

    # 3. Создание скоринговой функции
    scoring_function = get_scoring_function(activity_model)

    # 3.5. Дообучение моделей (если включено)
    if config.ENABLE_FINETUNING:
        LOGGER.info("=== Запуск дообучения моделей ===")
        try:
            from step_03_molecule_generation.run_finetuning import main as run_finetuning

            run_finetuning()
        except Exception as e:
            LOGGER.error(f"Fine-tuning failed: {e}")

    # 4. Генерация молекул согласно выбранному типу
    if config.GENERATOR_TYPE == "selfies_vae":
        LOGGER.info("Генерируем молекулы SELFIES-VAE…")
        from step_03_molecule_generation.selfies_vae_generator import train_and_sample

        generated_smiles = train_and_sample(config.VAE_GENERATE_N)
    elif config.GENERATOR_TYPE == "transformer_vae":
        LOGGER.info("Генерируем молекулы Transformer VAE…")
        from step_03_molecule_generation.transformer_vae_generator import train_and_sample_transformer

        generated_smiles = train_and_sample_transformer(config.VAE_GENERATE_N)

    elif config.GENERATOR_TYPE == "docking_guided":
        LOGGER.info("Генерируем молекулы с docking-guided RL…")
        from step_03_molecule_generation.char_rnn_generator import load_vocabulary  # Используем существующий vocabulary
        from step_03_molecule_generation.docking_guided_generator import DockingConfig, train_docking_guided_generator

        # Загружаем vocabulary
        vocab = load_vocabulary()

        # Создаем конфигурацию
        docking_config = DockingConfig(
            target_pdb=config.DOCKING_GUIDED_CONFIG["target_pdb"],
            chembl_id=config.DOCKING_GUIDED_CONFIG["chembl_id"],
            exhaustiveness=config.DOCKING_GUIDED_CONFIG["exhaustiveness"],
            num_modes=config.DOCKING_GUIDED_CONFIG["num_modes"],
            energy_range=config.DOCKING_GUIDED_CONFIG["energy_range"],
            docking_weight=config.DOCKING_GUIDED_CONFIG["docking_weight"],
            activity_weight=config.DOCKING_GUIDED_CONFIG["activity_weight"],
            drug_likeness_weight=config.DOCKING_GUIDED_CONFIG["drug_likeness_weight"],
            novelty_weight=config.DOCKING_GUIDED_CONFIG["novelty_weight"],
        )

        # Обучаем и генерируем
        model = train_docking_guided_generator(vocab, docking_config, num_epochs=config.DOCKING_GUIDED_CONFIG["rl_epochs"])

        # Генерируем финальные молекулы
        generated_smiles = model.generate_molecules(vocab, config.VAE_GENERATE_N)

    elif config.GENERATOR_TYPE == "graph_flow":
        LOGGER.info("Генерируем молекулы Graph Flow…")
        from step_03_molecule_generation.graph_generator import train_and_sample

        generated_smiles = train_and_sample(config.VAE_GENERATE_N)

    elif config.GENERATOR_TYPE == "pretrained":
        LOGGER.info("Генерируем молекулы с предобученной Hugging Face моделью…")
        from step_03_molecule_generation.pretrained_generator import PretrainedMolecularGenerator

        generator = PretrainedMolecularGenerator(
            model_name=config.PRETRAINED_HF_CONFIG["model_name"], max_length=config.PRETRAINED_HF_CONFIG["max_length"]
        )

        # Fine-tune if requested
        if config.PRETRAINED_HF_CONFIG["fine_tune"]:
            LOGGER.info("Дообучаем предобученную модель на целевых данных...")

            # Load ChEMBL data for fine-tuning
            if config.PRETRAINED_HF_CONFIG["use_chembl_data"]:
                try:
                    chembl_df = pl.read_parquet(config.ACTIVITY_DATA_PROCESSED_PATH)
                    target_smiles = chembl_df.select("SMILES").to_series().to_list()

                    # Sample if too large
                    sample_size = config.PRETRAINED_HF_CONFIG["chembl_sample_size"]
                    if len(target_smiles) > sample_size:
                        import random

                        random.seed(config.RANDOM_STATE)
                        target_smiles = random.sample(target_smiles, sample_size)

                    LOGGER.info(f"Дообучаем на {len(target_smiles)} молекулах из ChEMBL...")
                    generator.fine_tune_for_target(
                        target_smiles=target_smiles,
                        learning_rate=config.PRETRAINED_HF_CONFIG["fine_tune_lr"],
                        epochs=config.PRETRAINED_HF_CONFIG["fine_tune_epochs"],
                        batch_size=config.PRETRAINED_HF_CONFIG["fine_tune_batch_size"],
                    )
                except Exception as e:
                    LOGGER.warning(f"Не удалось загрузить данные ChEMBL для дообучения: {e}")

        # Generate molecules
        generated_smiles = generator.generate_molecules(
            num_molecules=config.PRETRAINED_HF_CONFIG["num_molecules"],
            temperature=config.PRETRAINED_HF_CONFIG["temperature"],
            top_k=config.PRETRAINED_HF_CONFIG["top_k"],
            top_p=config.PRETRAINED_HF_CONFIG["top_p"],
            batch_size=config.PRETRAINED_HF_CONFIG["batch_size"],
            filter_valid=config.PRETRAINED_HF_CONFIG["filter_valid"],
        )

        # Save raw SMILES for validation pipeline compatibility
        raw_smiles_path = config.GENERATION_RESULTS_DIR / "generated_smiles_raw.txt"
        with open(raw_smiles_path, "w", encoding="utf-8") as f:
            f.writelines(f"{smiles}\n" for smiles in generated_smiles)
        LOGGER.info(f"Raw SMILES saved to {raw_smiles_path}")

        # Evaluate molecules
        metrics = generator.evaluate_molecules(generated_smiles)
        LOGGER.info(f"Метрики генерации: {metrics}")

    else:
        raise ValueError(f"Unknown generator type: {config.GENERATOR_TYPE}")

    LOGGER.info(f"Генератор {config.GENERATOR_TYPE} сгенерировал {len(generated_smiles)} молекул")

    LOGGER.info(f"Оценка {len(generated_smiles)} сгенерированных молекул...")

    results = []
    for smiles in generated_smiles:
        mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
        if not mol:
            continue

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
            desc_vals = np.asarray(
                [
                    Descriptors.MolWt(mol),  # type: ignore[attr-defined]
                    Descriptors.MolLogP(mol),  # type: ignore[attr-defined]
                    Descriptors.TPSA(mol),  # type: ignore[attr-defined]
                    Descriptors.NumHDonors(mol),  # type: ignore[attr-defined]
                    Descriptors.NumHAcceptors(mol),  # type: ignore[attr-defined]
                    Descriptors.RingCount(mol),  # type: ignore[attr-defined]
                ],
                dtype=np.float32,
            )
            arr = np.concatenate([fp_arr, desc_vals])

        pIC50 = float(activity_model.predict(arr.reshape(1, -1))[0])

        results.append(
            {
                "smiles": smiles,
                "final_score": score,
                "predicted_pIC50": pIC50,
                "qed": QED.qed(mol),
                "logp": Crippen.MolLogP(mol),
                "sa_score": calculate_sa_score(smiles),
                "bbbp_prob": calculate_bbbp(smiles),
            }
        )

    generated_df = (
        pl.DataFrame(results)
        if results
        else pl.DataFrame(
            schema={
                "smiles": pl.Utf8,
                "final_score": pl.Float64,
                "predicted_pIC50": pl.Float64,
                "qed": pl.Float64,
                "logp": pl.Float64,
                "sa_score": pl.Float64,
                "bbbp_prob": pl.Float64,
            }
        )
    )

    generated_df.write_parquet(config.GENERATED_MOLECULES_PATH)

    LOGGER.info(f"Сгенерированные и оцененные молекулы сохранены в {config.GENERATED_MOLECULES_PATH}")
    if len(generated_df) == 0:
        LOGGER.warning("SELFIES-VAE не сгенерировал валидные молекулы – проверьте качество модели или увеличьте эпохи обучения.")
    else:
        top5 = generated_df.sort("final_score", descending=True).head(5)
        LOGGER.info(f"Топ-5 молекул по итоговому скору:\n{top5}")
    LOGGER.info("--- Этап 3 завершен ---")


if __name__ == "__main__":
    run_generation_pipeline()
