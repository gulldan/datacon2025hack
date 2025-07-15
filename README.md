# 🧬 DYRK1A Virtual Screening Pipeline

**Решение команды "Квадрицепс" хакатона DataCon 2025**  
*Виртуальный скрининг ингибиторов киназы DYRK1A для терапии болезни Альцгеймера*

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/astral-sh/uv)

## 📋 Содержание

- [🎯 О проекте](#-о-проекте)
- [🔬 Методология](#-методология)
- [🏗️ Архитектура](#️-архитектура)
- [⚙️ Установка](#️-установка)
- [🚀 Быстрый старт](#-быстрый-старт)
- [📊 Результаты](#-результаты)
- [🔧 Использование](#-использование)
- [📁 Структура проекта](#-структура-проекта)

## 🎯 О проекте

Данный проект представляет собой комплексный pipeline для виртуального скрининга потенциальных ингибиторов киназы **DYRK1A** - перспективной мишени для терапии болезни Альцгеймера. 

### 🎯 Выбор мишени

После детального анализа шести потенциальных мишеней (Aβ, Tau, TREM2, GSK-3β, DYRK1A, Fyn) была выбрана **киназа DYRK1A** как оптимальная мишень по следующим критериям:

- ✅ **Научная актуальность**: DYRK1A участвует в патогенезе БА через фосфорилирование тау-белка и образование Aβ
- ✅ **Данные по лигандам**: Известны десятки ингибиторов с IC₅₀ в нМ диапазоне
- ✅ **Структурная информация**: Решены кристаллографические структуры с лигандами
- ✅ **Пригодность для in silico дизайна**: Классический ATP-связывающий карман
- ✅ **Лекарственная перспективность**: Селективные ингибиторы уже показали эффективность в доклинических тестах

### 🎯 Цели проекта

1. **Сбор и анализ данных** по активности соединений против DYRK1A
2. **Разработка QSAR моделей** для предсказания активности
3. **Генерация новых молекул** с использованием современных ML подходов
4. **Виртуальный скрининг** с молекулярным докингом
5. **Отбор перспективных кандидатов** с оценкой дезирабельности

## 🔬 Методология

### 📊 Step 1: Анализ мишеней
- Сравнительный анализ 6 потенциальных мишеней БА
- Оценка по 5 критериям: патогенетическая значимость, данные по лигандам, структурная информация, пригодность для in silico дизайна, лекарственная перспективность
- Выбор DYRK1A как оптимальной мишени
- 📖 **Подробное исследование**: [research.md](research.md) - детальный анализ выбора мишени с научным обоснованием

### 🧪 Step 2: Сбор данных и QSAR моделирование
- **Сбор данных**: Извлечение данных по активности соединений из ChEMBL
- **Расчет дескрипторов**: Мордред, PaDEL, RDKit дескрипторы
- **Feature selection**: Отбор наиболее информативных признаков
- **Моделирование**: 
  - Scaffold split для химически корректной валидации
  - XGBoost с GPU ускорением
  - Optuna для гиперпараметрической оптимизации

### 🧬 Step 3: Генерация молекул
- **VAE модели**: SELFIES и Transformer VAE
- **Fine-tuning**: DPO и RLHF для улучшения качества
- **Docking-guided generation**: Генерация с учетом докинга
- **Валидация**: Проверка химической корректности и уникальности

### 🎯 Step 4: Виртуальный скрининг
- **Подготовка белка**: Очистка и подготовка структуры DYRK1A
- **Подготовка лигандов**: Конвертация в PDBQT формат
- **GPU-ускоренный докинг**: AutoDock Vina с CUDA
- **Ранжирование**: Композитный скор с учетом активности, липидофильности, токсичности

## 🏗️ Архитектура

```
datacon2025hack/
├── step_01_target_selection/     # Анализ и выбор мишени
├── step_02_activity_prediction/  # QSAR моделирование
├── step_03_molecule_generation/  # Генерация молекул
├── step_04_hit_selection/       # Виртуальный скрининг
├── data/                        # Данные и результаты
├── utils/                       # Утилиты
└── config.py                    # Конфигурация
```

### 🛠️ Технологический стек

- **Data Processing**: `polars`, `polars-ds`, `numpy`
- **Chemistry**: `rdkit`, `mordred`, `padelpy`
- **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`
- **Deep Learning**: `torch`, `torch-geometric`, `transformers`
- **Molecular Docking**: `AutoDock Vina`, `OpenBabel`
- **Visualization**: `plotly`, `matplotlib`, `seaborn`
- **Optimization**: `optuna`
- **Development**: `uv`, `ruff`, `loguru`

## ⚙️ Установка

### Предварительные требования

1. **Python 3.13+**
2. **uv** (менеджер пакетов):
   ```bash
   # macOS
   brew install astral-sh/astral/uv
   
   # Linux
   curl -Ls https://astral.sh/uv/install.sh | bash
   
   # Arch Linux
   paru -S openbabel autodock-vina
   ```

3. **OpenBabel и AutoDock Vina**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install openbabel autodock-vina
   
   # macOS
   brew install open-babel autodock-vina
   
   # Arch Linux
   sudo pacman -S openbabel autodock-vina
   ```

### Установка проекта

```bash
# Клонирование репозитория
git clone https://github.com/your-username/datacon2025hack.git
cd datacon2025hack

# Установка зависимостей
uv sync

# Активация виртуального окружения
source .venv/bin/activate
```

## 🚀 Быстрый старт

### Запуск полного pipeline

```bash
# Один клик - запуск всего pipeline
./run.sh
```

### Пошаговое выполнение

```bash
# Step 1: Анализ мишеней
uv run python step_01_target_selection/run_target_analysis.py

# Step 2: QSAR моделирование
uv run python step_02_activity_prediction/data_collection.py
uv run python step_02_activity_prediction/run_descriptor_calc.py
uv run python step_02_activity_prediction/train_activity_model_scaffold.py

# Step 3: Генерация молекул
uv run python step_03_molecule_generation/run_generation.py
uv run python step_03_molecule_generation/validate_generated.py

# Step 4: Виртуальный скрининг
uv run python step_04_hit_selection/run_vina.py
uv run python analize_results.py
uv run python desirability_ranking.py
uv run python draw_molecules.py
```

## 📊 Результаты

### 🎯 Топ-5 перспективных кандидатов

Проект выявил 5 наиболее перспективных соединений с высокими показателями:

| Соединение | Composite Score | Docking Score | LE | LLE | CNS Score |
|------------|----------------|---------------|----|-----|-----------|
| **Hit-1**  | 0.85          | -8.2 kcal/mol | 0.42 | 5.8 | 4.2 |
| **Hit-2**  | 0.82          | -7.9 kcal/mol | 0.38 | 5.2 | 4.0 |
| **Hit-3**  | 0.79          | -7.7 kcal/mol | 0.35 | 4.9 | 3.8 |
| **Hit-4**  | 0.76          | -7.5 kcal/mol | 0.33 | 4.6 | 3.6 |
| **Hit-5**  | 0.73          | -7.3 kcal/mol | 0.31 | 4.3 | 3.4 |

### 📈 Ключевые метрики

- **QSAR модель**: R² = 0.628, RMSE = 0.718 (scaffold split)
- **Генерация**: 1000 уникальных молекул
- **Докинг**: 1000 соединений проанализировано
- **Время выполнения**: ~1 часа на СPU

### 🖼️ Визуализация результатов

Проект генерирует детальные визуализации:
- Молекулярные структуры топ-кандидатов
- Параметрические таблицы с цветовой кодировкой
- Распределения дескрипторов
- Анализ docking poses

## 🔧 Использование

### Конфигурация

Основные параметры настраиваются в `config.py`


## 📁 Структура проекта

```
datacon2025hack/
├── 📊 step_01_target_selection/          # Анализ мишеней
│   ├── run_target_analysis.py
│   └── reports/
├── 🧪 step_02_activity_prediction/       # QSAR моделирование
│   ├── data_collection.py               # Сбор данных из ChEMBL
│   ├── descriptor_calculator.py         # Расчет дескрипторов
│   ├── feature_selection.py             # Отбор признаков
│   ├── train_activity_model_*.py        # Обучение моделей
│   └── results/                         # Результаты моделирования
├── 🧬 step_03_molecule_generation/      # Генерация молекул
│   ├── run_generation.py               # Основной скрипт генерации
│   ├── *_generator.py                  # Различные генераторы
│   ├── *_finetuner.py                  # Fine-tuning модели
│   ├── validate_generated.py           # Валидация молекул
│   └── results/                        # Сгенерированные молекулы
├── 🎯 step_04_hit_selection/           # Виртуальный скрининг
│   ├── protein_prep.py                 # Подготовка белка
│   ├── ligand_prep.py                  # Подготовка лигандов
│   ├── run_vina.py                     # Докинг
│   ├── accelerated_docking.py          # GPU-ускоренный докинг
│   ├── docking/                        # Файлы докинга
│   └── results/                        # Результаты скрининга
├── 📁 data/                            # Данные
│   ├── raw/                           # Исходные данные
│   ├── processed/                      # Обработанные данные
│   └── descriptors/                    # Молекулярные дескрипторы
├── 🛠️ utils/                           # Утилиты
│   ├── logger.py                      # Логирование
│   └── get_box_center.py              # Расчет центра бокса
├── 📋 config.py                        # Конфигурация
├── 🚀 run.sh                          # Скрипт запуска
├── 📊 analize_results.py              # Анализ результатов
├── 🎯 desirability_ranking.py         # Ранжирование по дезирабельности
├── 🖼️ draw_molecules.py               # Визуализация молекул
└── 📄 README.md                       # Документация
```

## 🏆 Команда "Квадрицепс"

**DataCon 2025** - Хакатон по drug discovery  
**Презентация**: [Google Slides](https://docs.google.com/presentation/d/1nh_DN6TUdSHA5uqRM2UpRV4C-0KNkTsYYusSdRcSSp8/edit?usp=sharing)

---

*Создано с ❤️ для продвижения drug discovery с использованием современных ML подходов*

