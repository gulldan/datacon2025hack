"""Reinforcement Learning from Human Feedback (RLHF) для молекулярных моделей.
Использует PPO (Proximal Policy Optimization) для дообучения на основе reward модели.
"""

import copy

# Import existing modules
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# RDKit imports
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, rdMolDescriptors
from torch import nn

sys.path.append("..")
from step_02_activity_prediction.model_utils import load_model, predict_activity
from utils.logger import LOGGER

logger = LOGGER

@dataclass
class RLHFConfig:
    """Конфигурация для RLHF обучения."""
    # PPO параметры
    learning_rate: float = 1e-5
    batch_size: int = 32
    ppo_epochs: int = 4
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # Общие параметры
    num_episodes: int = 100
    max_length: int = 80
    kl_penalty: float = 0.1

    # Reward модель
    reward_model_path: str | None = None
    target_pdb: str = "6S14"
    chembl_id: str = "CHEMBL3227"

    # Веса для reward функции
    docking_weight: float = 0.4
    activity_weight: float = 0.3
    qed_weight: float = 0.2
    sa_weight: float = 0.1

class RewardModel(nn.Module):
    """Модель для оценки качества молекул."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, molecular_features: torch.Tensor) -> torch.Tensor:
        """Предсказываем reward для молекулярных фичей."""
        return self.network(molecular_features)

class MolecularFeaturizer:
    """Извлекает молекулярные фичи для reward модели."""

    def __init__(self):
        self.activity_model = None
        self.load_activity_model()

    def load_activity_model(self):
        """Загружаем модель предсказания активности."""
        try:
            model_path = "../step_02_activity_prediction/results/activity_model_xgb.json"
            self.activity_model = load_model(model_path)
            logger.info("Loaded activity prediction model for reward calculation")
        except Exception as e:
            logger.error(f"Failed to load activity model: {e}")

    def smiles_to_features(self, smiles: str) -> np.ndarray:
        """Конвертируем SMILES в вектор фичей."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(2048)  # Возвращаем нулевой вектор для невалидных молекул

            features = []

            # 1. Базовые дескрипторы RDKit
            features.extend([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumSaturatedRings(mol),
                Descriptors.NumAliphaticRings(mol),
                QED.qed(mol)
            ])

            # 2. Дескрипторы синтетической доступности
            sa_score = rdMolDescriptors.BertzCT(mol)
            features.append(sa_score)

            # 3. Activity prediction
            if self.activity_model is not None:
                try:
                    activity_pred = predict_activity(self.activity_model, [smiles])
                    features.append(float(activity_pred[0]) if len(activity_pred) > 0 else 0.0)
                except:
                    features.append(0.0)
            else:
                features.append(0.0)

            # 4. Молекулярные фингерпринты (упрощенная версия)
            from rdkit.Chem import rdMolDescriptors
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_array = np.array(fp)

            # Объединяем все фичи
            basic_features = np.array(features)

            # Паддинг до 2048 размерности
            if len(basic_features) < 2048:
                padding = np.zeros(2048 - len(basic_features))
                combined_features = np.concatenate([basic_features, padding])
            else:
                combined_features = basic_features[:2048]

            return combined_features.astype(np.float32)

        except Exception as e:
            logger.debug(f"Feature extraction failed for {smiles}: {e}")
            return np.zeros(2048, dtype=np.float32)

class PPOTrainer:
    """PPO тренер для RLHF обучения."""

    def __init__(self, policy_model, value_model, reward_model, config: RLHFConfig):
        self.policy_model = policy_model
        self.value_model = value_model
        self.reward_model = reward_model
        self.config = config

        # Создаем копию для старой политики
        self.old_policy_model = copy.deepcopy(policy_model)

        # Оптимизаторы
        self.policy_optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=config.learning_rate
        )
        self.value_optimizer = torch.optim.AdamW(
            value_model.parameters(),
            lr=config.learning_rate
        )

        # Фичеризатор
        self.featurizer = MolecularFeaturizer()

    def calculate_reward(self, smiles: str) -> float:
        """Вычисляем reward для молекулы."""
        try:
            # Извлекаем фичи
            features = self.featurizer.smiles_to_features(smiles)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

            # Получаем reward от модели
            with torch.no_grad():
                reward = self.reward_model(features_tensor).item()

            return reward

        except Exception as e:
            logger.debug(f"Reward calculation failed for {smiles}: {e}")
            return -10.0  # Штраф за проблемные молекулы

    def generate_molecules(self, vocab, n_samples: int = 100) -> list[tuple[str, float]]:
        """Генерируем молекулы и вычисляем их rewards."""
        molecules_with_rewards = []

        # Простая генерация (нужна реальная реализация)
        for _ in range(n_samples):
            try:
                # Здесь должна быть реальная генерация из вашей модели
                # Пока используем заглушку
                sample_smiles = "CCO"  # Заглушка
                reward = self.calculate_reward(sample_smiles)
                molecules_with_rewards.append((sample_smiles, reward))
            except:
                continue

        return molecules_with_rewards

    def compute_advantages(self, rewards: list[float], values: list[float]) -> tuple[list[float], list[float]]:
        """Вычисляем advantages и returns для PPO."""
        returns = []
        advantages = []

        # Простое вычисление (можно улучшить с GAE)
        for i in range(len(rewards)):
            returns.append(rewards[i])
            advantages.append(rewards[i] - values[i])

        # Нормализация advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages.tolist(), returns

    def ppo_loss(self, states, actions, old_log_probs, advantages, returns):
        """Вычисляем PPO loss."""
        # Получаем новые log probabilities
        new_log_probs = self.policy_model.get_log_probs(states, actions)

        # Ratio для PPO
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        values = self.value_model(states)
        value_loss = F.mse_loss(values.squeeze(), returns)

        # Entropy bonus
        entropy = self.policy_model.get_entropy(states)
        entropy_loss = -entropy.mean()

        # Общий loss
        total_loss = (
            policy_loss +
            self.config.value_coef * value_loss +
            self.config.entropy_coef * entropy_loss
        )

        return total_loss, policy_loss, value_loss, entropy_loss

    def train_step(self, molecules_with_rewards: list[tuple[str, float]]):
        """Один шаг PPO обучения."""
        if len(molecules_with_rewards) < self.config.batch_size:
            return

        # Подготавливаем данные
        molecules = [mol for mol, _ in molecules_with_rewards]
        rewards = [reward for _, reward in molecules_with_rewards]

        # Получаем состояния (фичи молекул)
        states = []
        for mol in molecules:
            features = self.featurizer.smiles_to_features(mol)
            states.append(features)
        states = torch.tensor(states, dtype=torch.float32)

        # Получаем values
        with torch.no_grad():
            values = self.value_model(states).squeeze().tolist()

        # Вычисляем advantages и returns
        advantages, returns = self.compute_advantages(rewards, values)

        # Конвертируем в тензоры
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Получаем старые log probabilities
        with torch.no_grad():
            # Здесь нужна реальная реализация получения log_probs
            # old_log_probs = self.old_policy_model.get_log_probs(states, actions)
            old_log_probs = torch.zeros(len(molecules))  # Заглушка

        # PPO epochs
        for _ in range(self.config.ppo_epochs):
            # Здесь нужна реальная реализация PPO loss
            # loss, policy_loss, value_loss, entropy_loss = self.ppo_loss(
            #     states, actions, old_log_probs, advantages, returns
            # )

            # Заглушка для loss
            loss = torch.tensor(0.0, requires_grad=True)

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1.0)

            self.policy_optimizer.step()
            self.value_optimizer.step()

    def train(self, vocab, num_episodes: int):
        """Основной цикл RLHF обучения."""
        logger.info(f"Starting RLHF training for {num_episodes} episodes")

        for episode in range(num_episodes):
            # Генерируем молекулы
            molecules_with_rewards = self.generate_molecules(vocab, self.config.batch_size)

            if len(molecules_with_rewards) > 0:
                avg_reward = np.mean([reward for _, reward in molecules_with_rewards])
                logger.info(f"Episode {episode+1}, Average reward: {avg_reward:.4f}")

                # Обучаем модель
                self.train_step(molecules_with_rewards)

                # Обновляем старую политику
                if episode % 10 == 0:
                    self.old_policy_model.load_state_dict(self.policy_model.state_dict())

                # Сохраняем checkpoint
                if (episode + 1) % 20 == 0:
                    checkpoint_path = f"results/rlhf_model_episode_{episode+1}.pt"
                    torch.save({
                        "policy_model_state_dict": self.policy_model.state_dict(),
                        "value_model_state_dict": self.value_model.state_dict(),
                        "episode": episode,
                        "avg_reward": avg_reward
                    }, checkpoint_path)
                    logger.info(f"Saved checkpoint: {checkpoint_path}")

def train_reward_model(training_data: list[tuple[str, float]], config: RLHFConfig) -> RewardModel:
    """Обучаем reward модель на данных с человеческими предпочтениями."""
    logger.info("Training reward model...")

    featurizer = MolecularFeaturizer()
    reward_model = RewardModel()
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-4)

    # Подготавливаем данные
    features = []
    targets = []

    for smiles, score in training_data:
        feat = featurizer.smiles_to_features(smiles)
        features.append(feat)
        targets.append(score)

    features = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    # Обучение
    num_epochs = 50
    batch_size = 32

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for i in range(0, len(features), batch_size):
            batch_features = features[i:i + batch_size]
            batch_targets = targets[i:i + batch_size]

            optimizer.zero_grad()
            predictions = reward_model(batch_features).squeeze()
            loss = F.mse_loss(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        if (epoch + 1) % 10 == 0:
            logger.info(f"Reward model epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Сохраняем модель
    torch.save(reward_model.state_dict(), "results/reward_model.pt")
    logger.info("Reward model training completed")

    return reward_model

def finetune_with_rlhf(pretrained_model_path: str,
                      training_molecules: list[str],
                      config: RLHFConfig) -> nn.Module:
    """Основная функция для RLHF дообучения."""
    logger.info("Starting RLHF fine-tuning process")

    # 1. Загружаем предобученную модель
    from .dpo_finetuner import load_pretrained_model
    policy_model = load_pretrained_model(pretrained_model_path)

    # 2. Создаем value модель (может быть той же архитектуры)
    value_model = load_pretrained_model(pretrained_model_path)

    # 3. Создаем и обучаем reward модель
    # Создаем тренировочные данные с скорами
    featurizer = MolecularFeaturizer()
    training_data = []
    for smiles in training_molecules[:1000]:  # Ограничиваем для примера
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Простой скор на основе QED и активности
                qed_score = QED.qed(mol)
                score = qed_score  # Упрощенный скор
                training_data.append((smiles, score))
        except:
            continue

    reward_model = train_reward_model(training_data, config)

    # 4. Создаем vocabulary (нужна реальная реализация)
    vocab = {}  # Заглушка

    # 5. Создаем PPO тренер и обучаем
    trainer = PPOTrainer(policy_model, value_model, reward_model, config)
    trainer.train(vocab, config.num_episodes)

    logger.info("RLHF fine-tuning completed")
    return policy_model

if __name__ == "__main__":
    # Пример использования
    config = RLHFConfig(
        learning_rate=1e-5,
        batch_size=16,
        num_episodes=100
    )

    # Пример молекул для обучения
    example_molecules = [
        "CCO",
        "CC(=O)O",
        "c1ccccc1",
        # ... больше молекул
    ]

    # Дообучаем модель
    # finetuned_model = finetune_with_rlhf(
    #     "results/pretrained_model.pt",
    #     example_molecules,
    #     config
    # )

    logger.info("RLHF fine-tuning module ready")
