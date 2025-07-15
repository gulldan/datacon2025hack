"""Direct Preference Optimization (DPO) для дообучения молекулярных моделей.
Основано на статье: "Preference Optimization for Molecular Language Models" (arXiv:2310.12304)
"""

import json

# Import existing modules
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import selfies as sf
import torch
import torch.nn.functional as F

# RDKit imports
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from torch import nn
from tqdm import tqdm

sys.path.append("..")
from step_02_activity_prediction.model_utils import load_model, predict_activity
from utils.logger import LOGGER

logger = LOGGER


@dataclass
class DPOConfig:
    """Конфигурация для DPO дообучения."""

    beta: float = 0.1  # Temperature parameter for DPO
    learning_rate: float = 1e-5
    batch_size: int = 16
    num_epochs: int = 20
    max_length: int = 80
    reference_model_path: str | None = None
    target_pdb: str = "6S14"
    chembl_id: str = "CHEMBL3227"

    # Веса для создания предпочтений
    docking_weight: float = 0.4
    activity_weight: float = 0.3
    qed_weight: float = 0.2
    sa_weight: float = 0.1


class MolecularPreferenceDataset:
    """Датасет с предпочтениями для DPO обучения."""

    def __init__(self, config: DPOConfig):
        self.config = config
        self.preferences = []
        self.activity_model = None
        self.load_activity_model()

    def load_activity_model(self):
        """Загружаем модель предсказания активности."""
        try:
            model_path = "../step_02_activity_prediction/results/activity_model_xgb.json"
            self.activity_model = load_model(model_path)
            logger.info("Loaded activity prediction model for preference generation")
        except Exception as e:
            logger.error(f"Failed to load activity model: {e}")

    def calculate_molecule_score(self, smiles: str) -> float:
        """Вычисляем общий скор молекулы для ранжирования."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return -1000.0

            # 1. QED score
            qed_score = QED.qed(mol)

            # 2. SA Score (синтетическая доступность)
            from rdkit.Chem import rdMolDescriptors

            sa_score = rdMolDescriptors.BertzCT(mol) / 100.0  # Нормализуем
            sa_score = max(0, 1 - sa_score)  # Инвертируем (меньше = лучше)

            # 3. Activity prediction
            activity_score = 0.0
            if self.activity_model is not None:
                try:
                    activity_pred = predict_activity(self.activity_model, [smiles])
                    activity_score = float(activity_pred[0]) if len(activity_pred) > 0 else 0.0
                except:
                    activity_score = 0.0

            # 4. Простой docking proxy (липофильность как аппроксимация)
            logp = Descriptors.MolLogP(mol)
            docking_proxy = 1.0 / (1.0 + abs(logp - 2.5))  # Оптимальный LogP ~2.5

            # Комбинированный скор
            total_score = (
                self.config.qed_weight * qed_score
                + self.config.sa_weight * sa_score
                + self.config.activity_weight * activity_score
                + self.config.docking_weight * docking_proxy
            )

            return total_score

        except Exception as e:
            logger.debug(f"Score calculation failed for {smiles}: {e}")
            return -1000.0

    def create_preferences_from_molecules(self, molecules: list[str]) -> list[dict]:
        """Создаем пары предпочтений из списка молекул."""
        logger.info(f"Creating preferences from {len(molecules)} molecules...")

        # Вычисляем скоры для всех молекул
        scored_molecules = []
        for smiles in tqdm(molecules, desc="Scoring molecules"):
            score = self.calculate_molecule_score(smiles)
            if score > -100:  # Фильтруем совсем плохие молекулы
                scored_molecules.append((smiles, score))

        # Сортируем по скору
        scored_molecules.sort(key=lambda x: x[1], reverse=True)

        # Создаем пары предпочтений
        preferences = []
        n_pairs = min(1000, len(scored_molecules) // 2)  # Ограничиваем количество пар

        for i in range(n_pairs):
            # Берем хорошую молекулу из топа
            good_idx = np.random.randint(0, len(scored_molecules) // 4)
            # Берем плохую молекулу из низа
            bad_idx = np.random.randint(3 * len(scored_molecules) // 4, len(scored_molecules))

            good_smiles, good_score = scored_molecules[good_idx]
            bad_smiles, bad_score = scored_molecules[bad_idx]

            if good_score > bad_score:  # Проверяем, что предпочтение корректно
                preferences.append(
                    {"chosen": good_smiles, "rejected": bad_smiles, "chosen_score": good_score, "rejected_score": bad_score}
                )

        logger.info(f"Created {len(preferences)} preference pairs")
        return preferences

    def save_preferences(self, preferences: list[dict], path: str):
        """Сохраняем предпочтения в файл."""
        with open(path, "w") as f:
            json.dump(preferences, f, indent=2)
        logger.info(f"Saved {len(preferences)} preferences to {path}")

    def load_preferences(self, path: str) -> list[dict]:
        """Загружаем предпочтения из файла."""
        with open(path) as f:
            preferences = json.load(f)
        logger.info(f"Loaded {len(preferences)} preferences from {path}")
        return preferences


class DPOTrainer:
    """Тренер для DPO дообучения молекулярных моделей."""

    def __init__(self, model, reference_model, tokenizer, config: DPOConfig):
        self.model = model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.config = config

        # Замораживаем reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)

    def tokenize_molecules(self, molecules: list[str]) -> torch.Tensor:
        """Токенизируем молекулы в SELFIES."""
        tokenized = []
        for smiles in molecules:
            try:
                # Конвертируем в SELFIES
                selfies = sf.encoder(smiles)
                # Токенизируем (упрощенная версия)
                tokens = list(selfies)
                # Добавляем специальные токены
                tokens = ["<BOS>"] + tokens + ["<EOS>"]

                # Конвертируем в индексы (нужна реальная реализация токенизатора)
                # Здесь упрощенная версия
                token_ids = [self.tokenizer.get(token, 0) for token in tokens]

                # Паддинг до максимальной длины
                if len(token_ids) < self.config.max_length:
                    token_ids.extend([0] * (self.config.max_length - len(token_ids)))
                else:
                    token_ids = token_ids[: self.config.max_length]

                tokenized.append(token_ids)
            except:
                # Fallback для проблемных молекул
                tokenized.append([0] * self.config.max_length)

        return torch.tensor(tokenized, dtype=torch.long)

    def compute_dpo_loss(self, chosen_molecules: list[str], rejected_molecules: list[str]) -> torch.Tensor:
        """Вычисляем DPO loss."""
        # Токенизируем молекулы
        chosen_tokens = self.tokenize_molecules(chosen_molecules)
        rejected_tokens = self.tokenize_molecules(rejected_molecules)

        # Получаем логиты от основной модели
        chosen_logits = self.model(chosen_tokens)
        rejected_logits = self.model(rejected_tokens)

        # Получаем логиты от reference модели
        with torch.no_grad():
            chosen_ref_logits = self.reference_model(chosen_tokens)
            rejected_ref_logits = self.reference_model(rejected_tokens)

        # Вычисляем log probabilities
        chosen_logprobs = F.log_softmax(chosen_logits, dim=-1)
        rejected_logprobs = F.log_softmax(rejected_logits, dim=-1)
        chosen_ref_logprobs = F.log_softmax(chosen_ref_logits, dim=-1)
        rejected_ref_logprobs = F.log_softmax(rejected_ref_logits, dim=-1)

        # DPO loss
        chosen_rewards = chosen_logprobs - chosen_ref_logprobs
        rejected_rewards = rejected_logprobs - rejected_ref_logprobs

        # Усредняем по последовательности
        chosen_rewards = chosen_rewards.mean(dim=-1).mean(dim=-1)
        rejected_rewards = rejected_rewards.mean(dim=-1).mean(dim=-1)

        # DPO objective
        logits = self.config.beta * (chosen_rewards - rejected_rewards)
        loss = -F.logsigmoid(logits).mean()

        return loss

    def train_step(self, batch_preferences: list[dict]) -> float:
        """Один шаг обучения."""
        chosen_molecules = [pref["chosen"] for pref in batch_preferences]
        rejected_molecules = [pref["rejected"] for pref in batch_preferences]

        self.optimizer.zero_grad()
        loss = self.compute_dpo_loss(chosen_molecules, rejected_molecules)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        return loss.item()

    def train(self, preferences: list[dict]):
        """Основной цикл обучения."""
        logger.info(f"Starting DPO training with {len(preferences)} preferences")

        # Перемешиваем предпочтения
        np.random.shuffle(preferences)

        for epoch in range(self.config.num_epochs):
            total_loss = 0
            num_batches = 0

            # Батчи
            for i in range(0, len(preferences), self.config.batch_size):
                batch = preferences[i : i + self.config.batch_size]
                if len(batch) < self.config.batch_size:
                    continue

                loss = self.train_step(batch)
                total_loss += loss
                num_batches += 1

                if num_batches % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}, Batch {num_batches}, Loss: {loss:.4f}")

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch + 1} completed, Average loss: {avg_loss:.4f}")

            # Сохраняем checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"results/dpo_model_epoch_{epoch + 1}.pt"
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "loss": avg_loss,
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved checkpoint: {checkpoint_path}")


def load_pretrained_model(model_path: str):
    """Загружаем предобученную модель для дообучения."""
    # Здесь нужна реальная реализация загрузки вашей модели
    # Это может быть Transformer, VAE, или любая другая архитектура

    # Пример для Transformer модели
    from .transformer_vae_generator import TransformerVAE

    model = TransformerVAE(vocab_size=48)  # Ваш размер словаря

    if Path(model_path).exists():
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded pretrained model from {model_path}")
    else:
        logger.warning(f"Pretrained model not found at {model_path}, using random initialization")

    return model


def finetune_with_dpo(pretrained_model_path: str, generated_molecules: list[str], config: DPOConfig) -> nn.Module:
    """Основная функция для DPO дообучения."""
    logger.info("Starting DPO fine-tuning process")

    # 1. Загружаем предобученную модель
    model = load_pretrained_model(pretrained_model_path)

    # 2. Создаем reference модель (копия исходной)
    reference_model = load_pretrained_model(pretrained_model_path)

    # 3. Создаем простой токенизатор (нужна реальная реализация)
    tokenizer = {token: i for i, token in enumerate(["<PAD>", "<BOS>", "<EOS>"] + list("CNOS()[]=#-+"))}

    # 4. Создаем датасет с предпочтениями
    preference_dataset = MolecularPreferenceDataset(config)
    preferences = preference_dataset.create_preferences_from_molecules(generated_molecules)

    # Сохраняем предпочтения
    preference_dataset.save_preferences(preferences, "results/dpo_preferences.json")

    # 5. Создаем тренер и обучаем
    trainer = DPOTrainer(model, reference_model, tokenizer, config)
    trainer.train(preferences)

    logger.info("DPO fine-tuning completed")
    return model


if __name__ == "__main__":
    # Пример использования
    config = DPOConfig(beta=0.1, learning_rate=1e-5, batch_size=8, num_epochs=20)

    # Пример молекул для создания предпочтений
    example_molecules = [
        "CCO",  # Этанол
        "CC(=O)O",  # Уксусная кислота
        "c1ccccc1",  # Бензол
        # ... больше молекул
    ]

    # Дообучаем модель
    # finetuned_model = finetune_with_dpo(
    #     "results/pretrained_model.pt",
    #     example_molecules,
    #     config
    # )

    logger.info("DPO fine-tuning module ready")
