"""Основной скрипт для дообучения готовых молекулярных моделей.
Поддерживает DPO и RLHF методы дообучения.
"""

import sys

sys.path.append("..")


import polars as pl

import config
from utils.logger import LOGGER

logger = LOGGER


def load_existing_molecules() -> list[str]:
    """Загружаем существующие сгенерированные молекулы для дообучения."""
    molecules = []

    # 1. Загружаем из результатов генерации
    if config.GENERATED_MOLECULES_PATH.exists():
        logger.info(f"Loading molecules from {config.GENERATED_MOLECULES_PATH}")
        df = pl.read_parquet(config.GENERATED_MOLECULES_PATH)
        if "smiles" in df.columns:
            molecules.extend(df["smiles"].to_list())

    # 2. Загружаем из обучающих данных CHEMBL
    if config.ACTIVITY_DATA_PROCESSED_PATH.exists():
        logger.info(f"Loading training molecules from {config.ACTIVITY_DATA_PROCESSED_PATH}")
        df = pl.read_parquet(config.ACTIVITY_DATA_PROCESSED_PATH)
        if "SMILES" in df.columns:
            training_molecules = df["SMILES"].to_list()
            molecules.extend(training_molecules[:1000])  # Ограничиваем количество

    # Удаляем дубликаты
    molecules = list(set(molecules))
    logger.info(f"Loaded {len(molecules)} unique molecules for fine-tuning")

    return molecules


def run_dpo_finetuning(pretrained_model_path: str, molecules: list[str]) -> str:
    """Запускаем DPO дообучение."""
    logger.info("Starting DPO fine-tuning...")

    from .dpo_finetuner import DPOConfig, finetune_with_dpo

    # Создаем конфигурацию
    dpo_config = DPOConfig(
        beta=config.DPO_CONFIG["beta"],
        learning_rate=config.DPO_CONFIG["learning_rate"],
        batch_size=config.DPO_CONFIG["batch_size"],
        num_epochs=config.DPO_CONFIG["num_epochs"],
        max_length=config.DPO_CONFIG["max_length"],
        target_pdb=config.DPO_CONFIG["target_pdb"],
        chembl_id=config.DPO_CONFIG["chembl_id"],
        docking_weight=config.DPO_CONFIG["docking_weight"],
        activity_weight=config.DPO_CONFIG["activity_weight"],
        qed_weight=config.DPO_CONFIG["qed_weight"],
        sa_weight=config.DPO_CONFIG["sa_weight"],
    )

    # Дообучаем модель
    finetuned_model = finetune_with_dpo(pretrained_model_path, molecules, dpo_config)

    # Сохраняем дообученную модель
    output_path = config.GENERATION_RESULTS_DIR / "dpo_finetuned_model.pt"
    torch.save({"model_state_dict": finetuned_model.state_dict(), "config": dpo_config, "method": "dpo"}, output_path)

    logger.info(f"DPO fine-tuned model saved to {output_path}")
    return str(output_path)


def run_rlhf_finetuning(pretrained_model_path: str, molecules: list[str]) -> str:
    """Запускаем RLHF дообучение."""
    logger.info("Starting RLHF fine-tuning...")

    from .rlhf_finetuner import RLHFConfig, finetune_with_rlhf

    # Создаем конфигурацию
    rlhf_config = RLHFConfig(
        learning_rate=config.RLHF_CONFIG["learning_rate"],
        batch_size=config.RLHF_CONFIG["batch_size"],
        ppo_epochs=config.RLHF_CONFIG["ppo_epochs"],
        clip_ratio=config.RLHF_CONFIG["clip_ratio"],
        value_coef=config.RLHF_CONFIG["value_coef"],
        entropy_coef=config.RLHF_CONFIG["entropy_coef"],
        num_episodes=config.RLHF_CONFIG["num_episodes"],
        max_length=config.RLHF_CONFIG["max_length"],
        kl_penalty=config.RLHF_CONFIG["kl_penalty"],
        target_pdb=config.RLHF_CONFIG["target_pdb"],
        chembl_id=config.RLHF_CONFIG["chembl_id"],
        docking_weight=config.RLHF_CONFIG["docking_weight"],
        activity_weight=config.RLHF_CONFIG["activity_weight"],
        qed_weight=config.RLHF_CONFIG["qed_weight"],
        sa_weight=config.RLHF_CONFIG["sa_weight"],
    )

    # Дообучаем модель
    finetuned_model = finetune_with_rlhf(pretrained_model_path, molecules, rlhf_config)

    # Сохраняем дообученную модель
    output_path = config.GENERATION_RESULTS_DIR / "rlhf_finetuned_model.pt"
    torch.save({"model_state_dict": finetuned_model.state_dict(), "config": rlhf_config, "method": "rlhf"}, output_path)

    logger.info(f"RLHF fine-tuned model saved to {output_path}")
    return str(output_path)


def generate_with_finetuned_model(model_path: str, n_samples: int = 1000) -> list[str]:
    """Генерируем молекулы с помощью дообученной модели."""
    logger.info(f"Generating {n_samples} molecules with fine-tuned model...")

    import torch

    # Загружаем модель
    checkpoint = torch.load(model_path)
    method = checkpoint["method"]

    logger.info(f"Using {method.upper()} fine-tuned model")

    if method == "dpo":
        from .dpo_finetuner import load_pretrained_model

        model = load_pretrained_model(model_path)
    elif method == "rlhf":
        from .rlhf_finetuner import load_pretrained_model

        model = load_pretrained_model(model_path)
    else:
        raise ValueError(f"Unknown fine-tuning method: {method}")

    # Загружаем модель состояние
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Генерируем молекулы (нужна реальная реализация)
    generated_molecules = []

    # Здесь должна быть реальная генерация
    # Пока используем заглушку
    logger.warning("Using placeholder generation - implement real generation logic")

    return generated_molecules


def main():
    """Основная функция для запуска дообучения."""
    logger.info("=== Starting Fine-tuning Pipeline ===")

    if not config.ENABLE_FINETUNING:
        logger.info("Fine-tuning is disabled in config")
        return

    # 1. Загружаем молекулы для дообучения
    molecules = load_existing_molecules()

    if len(molecules) < 100:
        logger.warning(f"Only {len(molecules)} molecules available for fine-tuning. Consider generating more first.")
        return

    # 2. Определяем путь к предобученной модели
    pretrained_model_path = None

    # Пробуем найти существующую модель
    for model_type, path in config.PRETRAINED_MODEL_PATHS.items():
        if path.exists():
            pretrained_model_path = str(path)
            logger.info(f"Found pretrained model: {model_type} at {path}")
            break

    if pretrained_model_path is None:
        logger.error("No pretrained model found. Please train a model first.")
        return

    finetuned_models = []

    # 3. Запускаем дообучение
    if config.FINETUNING_METHOD in ["dpo", "both"]:
        try:
            dpo_model_path = run_dpo_finetuning(pretrained_model_path, molecules)
            finetuned_models.append(("DPO", dpo_model_path))
        except Exception as e:
            logger.error(f"DPO fine-tuning failed: {e}")

    if config.FINETUNING_METHOD in ["rlhf", "both"]:
        try:
            rlhf_model_path = run_rlhf_finetuning(pretrained_model_path, molecules)
            finetuned_models.append(("RLHF", rlhf_model_path))
        except Exception as e:
            logger.error(f"RLHF fine-tuning failed: {e}")

    # 4. Генерируем молекулы с дообученными моделями
    for method_name, model_path in finetuned_models:
        logger.info(f"Generating molecules with {method_name} model...")
        try:
            generated_molecules = generate_with_finetuned_model(model_path, config.VAE_GENERATE_N)

            if generated_molecules:
                # Сохраняем результаты
                output_path = config.GENERATION_RESULTS_DIR / f"{method_name.lower()}_generated_molecules.parquet"
                df = pl.DataFrame({"smiles": generated_molecules})
                df.write_parquet(output_path)
                logger.info(f"Saved {len(generated_molecules)} molecules to {output_path}")
            else:
                logger.warning(f"No molecules generated with {method_name} model")

        except Exception as e:
            logger.error(f"Generation failed with {method_name} model: {e}")

    logger.info("=== Fine-tuning Pipeline Completed ===")


if __name__ == "__main__":
    import torch  # Добавляем импорт здесь

    main()
