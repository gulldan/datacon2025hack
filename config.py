# config.py
from pathlib import Path

# --- Основные пути ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Создание директорий, если они не существуют
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Шаг 1: Выбор мишени ---
TARGET_SELECTION_DIR = BASE_DIR / "step_01_target_selection"
TARGET_REPORTS_DIR = TARGET_SELECTION_DIR / "reports"
TARGET_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Выбранная мишень
# CHEMBL ID для DYRK1A человека: CHEMBL3227
# PDB ID для структуры с лигандом: 6S14
CHOSEN_TARGET_ID = "CHEMBL3227"
CHOSEN_PDB_ID = "6S14"

# --- DYRK1A специфичные параметры для болезни Альцгеймера ---
# Согласно исследованиям, DYRK1A является ключевой мишенью при болезни Альцгеймера
# Источник: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3503344/
DYRK1A_ALZHEIMER_CONFIG = {
    "target_name": "DYRK1A",
    "disease": "Alzheimer's Disease",
    "mechanism": "Tau hyperphosphorylation and amyloid precursor protein processing",
    "therapeutic_rationale": "Inhibition reduces tau pathology and amyloid production",

    # Активность - для DYRK1A при болезни Альцгеймера нужна высокая селективность
    "activity_thresholds": {
        "high_activity": 7.0,      # pIC50 > 7.0 (IC50 < 100 nM) - высокая активность
        "moderate_activity": 6.0,   # pIC50 > 6.0 (IC50 < 1 μM) - умеренная активность
        "low_activity": 5.0,       # pIC50 > 5.0 (IC50 < 10 μM) - низкая активность
        "inactive": 4.0            # pIC50 < 4.0 (IC50 > 100 μM) - неактивные
    },

    # Селективность - важно для избежания побочных эффектов
    "selectivity_targets": [
        "DYRK1B",  # Близкий гомолог
        "DYRK2",   # Семейство DYRK
        "GSK3B",   # Участвует в том же пути
        "CDK5",    # Также фосфорилирует tau
        "CK1"      # Киназа tau
    ],

    # Профиль безопасности для ЦНС
    "safety_profile": {
        "bbb_permeability": 0.7,    # Минимальная проницаемость ГЭБ
        "neurotoxicity_risk": 0.3,  # Максимальный риск нейротоксичности
        "cardiotoxicity_risk": 0.2, # Максимальный риск кардиотоксичности
        "hepatotoxicity_risk": 0.3  # Максимальный риск гепатотоксичности
    }
}

# --- Шаг 2: Предсказание активности ---
PREDICTION_DIR = BASE_DIR / "step_02_activity_prediction"
PREDICTION_RESULTS_DIR = PREDICTION_DIR / "results"
PREDICTION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITY_DATA_RAW_PATH = RAW_DATA_DIR / f"{CHOSEN_TARGET_ID}_raw.parquet"
ACTIVITY_DATA_PROCESSED_PATH = PROCESSED_DATA_DIR / f"{CHOSEN_TARGET_ID}_processed.parquet"
MODEL_PATH = PREDICTION_RESULTS_DIR / "activity_model.npz"
# Альтернативная, более сложная модель (XGBoost)
XGB_MODEL_PATH = PREDICTION_RESULTS_DIR / "activity_model_xgb.json"
EDA_PLOTS_PATH = PREDICTION_RESULTS_DIR / "eda_plots.html"
FEATURE_IMPORTANCE_PATH = PREDICTION_RESULTS_DIR / "feature_importance.html"

# --- Шаг 2: Параметры модели активности и отпечатков ---
# Настройки Morgan-отпечатков, используемых при обучении линейных
# и градиентных моделей (ElasticNet, XGBoost) на этапе 2.
FP_RADIUS = 2                       # радиус окружения атома
FP_BITS_LINEAR = 2048               # длина отпечатка для линейной модели
FP_BITS_XGB = 1024                  # длина отпечатка для XGBoost
FP_INCLUDE_CHIRALITY = False        # учитывать ли хиральность

# Гиперпараметры XGBoost-модели активности
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.4,
    # Быстрый histogram-бэкэнд с поддержкой GPU (treelite)
    "tree_method": "hist",
    "device": "cuda",
}
XGB_NUM_BOOST_ROUND = 1000          # число итераций бустинга
XGB_EARLY_STOPPING_ROUNDS = 100     # early-stopping на валидации

# --- Шаг 3: Генерация молекул ---
GENERATION_DIR = BASE_DIR / "step_03_molecule_generation"
GENERATION_RESULTS_DIR = GENERATION_DIR / "results"
GENERATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GENERATED_MOLECULES_PATH = GENERATION_RESULTS_DIR / "generated_molecules.parquet"

# --- Шаг 3: Конфигурация генераторов ---
# Выбор генеративной модели:
# "selfies_vae" - базовый RNN VAE (текущий)
# "transformer_vae" - Transformer VAE (улучшенный)
# "docking_guided" - Docking-guided RL генератор (новый)
# "pretrained" - Предобученная модель с Hugging Face (новый)
# "graph_flow" - Graph Flow (будущий T21)
GENERATOR_TYPE = "pretrained"  # Переключаем на исправленный Transformer VAE
# Путь к сохранённой модели генерации графов (если выбран graph_flow)
GRAPH_FLOW_MODEL_PATH = GENERATION_RESULTS_DIR / "graph_flow.pt"

# --- Шаг 3: Гиперпараметры генеративной SELFIES-VAE и скоринга ---
# Основные размеры сети (уменьшаем для предотвращения переобучения)
VAE_EMBED_DIM = 128
VAE_HIDDEN_DIM = 256
VAE_LATENT_DIM = 64
VAE_NUM_LAYERS = 2
VAE_DROPOUT = 0.3              # Увеличиваем dropout для регуляризации
# Обучение
VAE_BATCH_SIZE = 32            # Уменьшаем batch size для более стабильного обучения
VAE_MAX_LEN = 80               # Сокращаем максимальную длину
VAE_LEARNING_RATE = 5e-4       # Уменьшаем learning rate
# Увеличиваем patience для предотвращения преждевременной остановки
VAE_PATIENCE = 15
# Максимальное число эпох (можно переопределить переменной окружения)
MAX_VAE_EPOCHS = 50            # Reduced from 200 with better early stopping
# Выборка после обучения
VAE_GENERATE_N = 3000          # Генерируем меньше, но качественнее
# Максимальный размер батча при семплинге (GPU память)
VAE_SAMPLE_BATCH = 512

# Аугментация данных (рандомные SMILES на молекулу)
AUG_PER_MOL = 10

# VAE Annealing Schedule Configuration - Fixes KL Vanishing
VAE_ANNEALING_TYPE = "cyclical"  # "cyclical", "monotonic", "logistic", "constant"
VAE_ANNEALING_CYCLES = 6         # Увеличиваем количество циклов для лучшего обучения
VAE_ANNEALING_RATIO = 0.4        # Уменьшаем долю роста β для более плавного обучения
VAE_MAX_BETA = 0.05              # Уменьшаем максимальный β для предотвращения коллапса

# --- Docking-guided generation settings ---
DOCKING_GUIDED_CONFIG = {
    "target_pdb": CHOSEN_PDB_ID,
    "chembl_id": CHOSEN_TARGET_ID,
    "exhaustiveness": 8,
    "num_modes": 9,
    "energy_range": 3.0,
    "docking_weight": 0.4,
    "activity_weight": 0.3,
    "drug_likeness_weight": 0.2,
    "novelty_weight": 0.1,
    "rl_epochs": 50,
    "rl_batch_size": 32,
    "max_length": 80,
    "learning_rate": 1e-4
}

# --- Fine-tuning configurations ---
# Включить дообучение готовых моделей
ENABLE_FINETUNING = True
FINETUNING_METHOD = "dpo"  # "dpo", "rlhf", "both"

# DPO (Direct Preference Optimization) settings
DPO_CONFIG = {
    "beta": 0.1,
    "learning_rate": 1e-5,
    "batch_size": 16,
    "num_epochs": 20,
    "max_length": 80,
    "target_pdb": CHOSEN_PDB_ID,
    "chembl_id": CHOSEN_TARGET_ID,
    "docking_weight": 0.4,
    "activity_weight": 0.3,
    "qed_weight": 0.2,
    "sa_weight": 0.1
}

# RLHF (Reinforcement Learning from Human Feedback) settings
RLHF_CONFIG = {
    "learning_rate": 1e-5,
    "batch_size": 32,
    "ppo_epochs": 4,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "num_episodes": 100,
    "max_length": 80,
    "kl_penalty": 0.1,
    "target_pdb": CHOSEN_PDB_ID,
    "chembl_id": CHOSEN_TARGET_ID,
    "docking_weight": 0.4,
    "activity_weight": 0.3,
    "qed_weight": 0.2,
    "sa_weight": 0.1
}

# Пути для предобученных моделей
PRETRAINED_MODEL_PATHS = {
    "transformer_vae": GENERATION_RESULTS_DIR / "transformer_vae.pt",
    "selfies_vae": GENERATION_RESULTS_DIR / "selfies_vae.pt",
    "char_rnn": GENERATION_RESULTS_DIR / "char_rnn.pt"
}

# Конфигурация предобученных моделей Hugging Face
PRETRAINED_HF_CONFIG = {
    "model_name": "entropy/gpt2_zinc_87m",  # Основная модель
    "alternative_models": [
        "seyonec/ChemBERTa-zinc-base-v1",
        "seyonec/PubChem10M_SMILES_BPE_60k",
        "DeepChem/SmilesTokenizer_PubChem_1M"
    ],
    "max_length": 256,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.95,
    "batch_size": 32,
    "num_molecules": 10000,
    "filter_valid": True,
    "fine_tune": True,
    "fine_tune_epochs": 3,
    "fine_tune_lr": 1e-5,
    "fine_tune_batch_size": 16,
    "use_chembl_data": True,  # Использовать данные ChEMBL для fine-tuning
    "chembl_sample_size": 1000  # Размер выборки из ChEMBL
}

# --- Молекулярные дескрипторы и фильтры для DYRK1A ---
# Основанные на исследованиях DYRK1A ингибиторов для болезни Альцгеймера
MOLECULAR_DESCRIPTORS_CONFIG = {
    # Lipinski's Rule of Five - базовые фильтры
    "molecular_weight": {
        "min": 150.0,     # Минимальная MW для активности
        "max": 500.0,     # Максимальная MW для проницаемости ГЭБ
        "optimal": 350.0  # Оптимальная MW для ЦНС препаратов
    },

    # LogP - важно для проницаемости ГЭБ
    "logp": {
        "min": 1.0,       # Минимальный LogP для активности
        "max": 4.0,       # Максимальный LogP для растворимости
        "optimal": 2.5    # Оптимальный LogP для ЦНС
    },

    # Полярная поверхность - критично для ГЭБ
    "tpsa": {
        "min": 20.0,      # Минимальная TPSA
        "max": 90.0,      # Максимальная TPSA для ГЭБ (обычно <140, но для ЦНС строже)
        "optimal": 60.0   # Оптимальная TPSA для ЦНС
    },

    # Водородные связи
    "hbd": {
        "max": 3          # Максимальное количество донорных групп
    },

    "hba": {
        "max": 7          # Максимальное количество акцепторных групп
    },

    # Ротационные связи - влияют на связывание
    "rotatable_bonds": {
        "max": 10         # Максимальное количество ротационных связей
    },

    # Ароматические кольца - важны для связывания с DYRK1A
    "aromatic_rings": {
        "min": 1,         # Минимальное количество ароматических колец
        "max": 4,         # Максимальное количество
        "optimal": 2      # Оптимальное количество
    }
}

# --- Фильтры для отбора хитов DYRK1A ---
HIT_SELECTION_FILTERS = {
    # Активность - основанная на литературных данных для DYRK1A
    "activity_filters": {
        "predicted_pic50": {
            "min": 5.0,       # Минимальная активность (IC50 < 10 μM)
            "good": 6.0,      # Хорошая активность (IC50 < 1 μM)
            "excellent": 7.0  # Отличная активность (IC50 < 100 nM)
        }
    },

    # Drug-likeness фильтры
    "drug_likeness_filters": {
        "qed": {
            "min": 0.3,       # Минимальный QED
            "good": 0.5,      # Хороший QED
            "excellent": 0.7  # Отличный QED
        }
    },

    # Синтетическая доступность
    "synthetic_accessibility": {
        "sa_score": {
            "max": 6.0,       # Максимальный SA score (более мягкий для heuristic)
            "good": 4.0,      # Хороший SA score
            "excellent": 3.0  # Отличный SA score
        }
    },

    # Проницаемость ГЭБ - критично для лечения болезни Альцгеймера
    "bbb_permeability": {
        "min": 0.3,          # Минимальная вероятность проницаемости
        "good": 0.5,         # Хорошая проницаемость
        "excellent": 0.7     # Отличная проницаемость
    },

    # Докинг - энергия связывания с DYRK1A
    "docking_filters": {
        "binding_energy": {
            "max": -6.0,      # Максимальная энергия связывания (более отрицательная = лучше)
            "good": -7.0,     # Хорошая энергия связывания
            "excellent": -8.0 # Отличная энергия связывания
        }
    },

    # Селективность - важно для избежания побочных эффектов
    "selectivity_filters": {
        "min_selectivity_ratio": 10.0,  # Минимальное соотношение селективности
        "target_kinases": [
            "DYRK1B", "DYRK2", "GSK3B", "CDK5", "CK1"
        ]
    }
}

# Весовые коэффициенты финального скоринга генерации (сумма = 1.0)
# Оптимизированы для DYRK1A и болезни Альцгеймера
SCORING_WEIGHTS = {
    "activity": 0.35,      # Активность против DYRK1A
    "qed": 0.20,          # Drug-likeness
    "sa": 0.15,           # Синтетическая доступность
    "bbbp": 0.25,         # Проницаемость ГЭБ (повышена для ЦНС)
    "selectivity": 0.05   # Селективность
}

# --- ADMET фильтры ---
USE_CYP450_FILTERS = True                # применять ли CYP450 фильтр
CYP450_ISOFORMS = ["1A2", "2C9", "2C19", "2D6", "3A4"]  # ключевые изоферменты
# BRENK / токсофоры фильтр (T30)
USE_BRENK_FILTER = True                  # применять ли набор BRENK substructure filters
USE_HEPATOTOX_FILTER = True              # фильтр потенциальной гепатотоксичности

# --- Специфичные для ЦНС ADMET фильтры ---
CNS_ADMET_FILTERS = {
    "blood_brain_barrier": {
        "min_permeability": 0.3,    # Минимальная проницаемость ГЭБ
        "use_egan_rule": True,      # Использовать правило Egan (TPSA ≤ 132, LogP ≤ 5.9)
        "use_veber_rule": True      # Использовать правило Veber (RotBonds ≤ 10, TPSA ≤ 140)
    },

    "neurotoxicity": {
        "max_risk": 0.3,           # Максимальный риск нейротоксичности
        "check_off_targets": [     # Проверка на нежелательные мишени
            "HERG", "NAV1.5", "CACNA1C", "KCNQ1"
        ]
    },

    "metabolic_stability": {
        "min_half_life": 2.0,      # Минимальный период полувыведения (часы)
        "max_clearance": 50.0      # Максимальный клиренс (mL/min/kg)
    }
}

# --- Optuna автоматический подбор гиперпараметров ---
OPTUNA_TUNE_XGB = False                  # вкл/выкл поиск для XGBoost
OPTUNA_TUNE_VAE = False                  # вкл/выкл поиск для VAE/генераторов
OPTUNA_STUDIES_DIR = BASE_DIR / "optuna_studies"
OPTUNA_STUDIES_DIR.mkdir(parents=True, exist_ok=True)

# --- Docking parameters ---
# Позволяет отключить реальный запуск AutoDock Vina (например, если не установлен)
USE_VINA_DOCKING = True

# --- Выбор режима докинга ---
# Параметр для выбора между CPU и GPU докингом в pipeline
DOCKING_MODE = "cpu"  # "cpu", "gpu"
# "cpu" - использовать только CPU Vina
# "gpu" - использовать GPU AutoDock-GPU с автоматическим fallback на CPU

# --- Шаг 4: Отбор хитов ---
HIT_SELECTION_DIR = BASE_DIR / "step_04_hit_selection"
HIT_SELECTION_RESULTS_DIR = HIT_SELECTION_DIR / "results"
HIT_SELECTION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_HITS_PATH = HIT_SELECTION_RESULTS_DIR / "final_hits.parquet"
DOCKING_RESULTS_PATH = HIT_SELECTION_RESULTS_DIR / "docking_scores.parquet"

DOCKING_DIR = HIT_SELECTION_DIR / "docking"
DOCKING_DIR.mkdir(parents=True, exist_ok=True)

# Raw PDB of chosen target
PROTEIN_PDB_PATH = DOCKING_DIR / f"{CHOSEN_PDB_ID}.pdb"
# Prepared receptor for AutoDock Vina
PROTEIN_PDBQT_PATH = DOCKING_DIR / f"{CHOSEN_PDB_ID}_receptor.pdbqt"
# Directory for ligand PDBQT files
LIGAND_PDBQT_DIR = DOCKING_DIR / "ligands"
LIGAND_PDBQT_DIR.mkdir(parents=True, exist_ok=True)
# Docking poses/scores
VINA_RESULTS_PATH = DOCKING_DIR / "vina_scores.parquet"

# --- Docking box parameters для DYRK1A (PDB: 6S14) ---
# Координаты активного сайта DYRK1A, оптимизированные для связывания ингибиторов
# Основаны на структурных исследованиях DYRK1A ингибиторов
BOX_CENTER = (16.5, 9.8, 25.7)     # Центр grid box (Å) - активный сайт DYRK1A
BOX_SIZE = (20.0, 20.0, 20.0)      # Размер grid box (Å) - достаточно для покрытия сайта связывания

# --- Дополнительные параметры докинга для DYRK1A ---
DOCKING_PARAMETERS = {
    "exhaustiveness": 8,         # Тщательность поиска
    "num_modes": 9,             # Количество режимов связывания
    "energy_range": 3.0,        # Диапазон энергий (kcal/mol)

    # GPU докинг настройки - ОПТИМИЗИРОВАНЫ ДЛЯ МАКСИМАЛЬНОЙ ЗАГРУЗКИ GPU
    "gpu_engine": "autodock_gpu",
    "use_gpu": True,
    "autodock_gpu_path": "/home/qwerty/github/datacon2025hack/gpu_docking_tools/AutoDock-GPU-develop/bin/autodock_gpu_128wi",
    "autogrid_path": "/usr/local/bin/autogrid4",

    # Оптимизированные параметры для максимальной загрузки GPU
    "batch_size": 2000,          # Увеличиваем размер батча для лучшей загрузки GPU
    "max_concurrent_jobs": 16,   # Больше параллельных задач для GPU
    "gpu_device": 0,             # Основной GPU
    "num_threads": 32,           # Максимальное количество потоков

    # AutoDock-GPU специфичные параметры для высокой производительности
    "autodock_gpu_nrun": 50,     # Увеличиваем количество запусков для лучшей загрузки
    "autodock_gpu_nev": 5000000, # Больше оценок для интенсивной GPU работы
    "autodock_gpu_ngen": 84000,  # Увеличиваем поколения для длительной GPU работы
    "autodock_gpu_psize": 300,   # Увеличиваем размер популяции
    "autodock_gpu_heuristics": 1,
    "autodock_gpu_autostop": 0,  # Отключаем автостоп для максимальной загрузки
    "autodock_gpu_xml_output": 1,
    "autodock_gpu_dlg_output": 1,

    # Таймауты для длительных GPU вычислений
    "timeout_per_ligand": 600,   # Увеличиваем таймаут для сложных вычислений
    "timeout_per_batch": 3600,   # Таймаут для батча

    # Ключевые остатки для взаимодействия с DYRK1A
    "key_residues": [
        "LYS188",  # Консервативный лизин
        "GLU239",  # Hinge region
        "LEU241",  # Hinge region
        "PHE238"   # Gatekeeper residue
    ],

    # Фармакофорные особенности для DYRK1A
    "pharmacophore_features": [
        "hinge_binding",        # Связывание с hinge region
        "atp_binding_site",     # Связывание с ATP-сайтом
        "selectivity_pocket"    # Карман селективности
    ]
}

# --- PaDEL Descriptor ---
# Путь к PaDEL-Descriptor.jar (скачайте с https://github.com/dataprofessor/padel)
PADEL_JAR_PATH = BASE_DIR / "external" / "PaDEL-Descriptor.jar"
USE_PADEL_DESCRIPTORS = False  # установить True, если Java и PaDEL.jar доступны

# --- Параметры моделей ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Параметры для различных этапов pipeline ---
PIPELINE_PARAMETERS = {
    "data_preprocessing": {
        "remove_duplicates": True,
        "standardize_smiles": True,
        "filter_invalid": True,
        "min_heavy_atoms": 6,
        "max_heavy_atoms": 50
    },

    "feature_generation": {
        "fingerprint_type": "morgan",
        "radius": FP_RADIUS,
        "n_bits": FP_BITS_LINEAR,
        "use_features": True,
        "use_chirality": FP_INCLUDE_CHIRALITY
    },

    "model_training": {
        "cross_validation_folds": 5,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratify": True
    },

    "molecule_generation": {
        "batch_size": VAE_BATCH_SIZE,
        "max_length": VAE_MAX_LEN,
        "temperature": 1.0,
        "diversity_penalty": 0.1
    },

    "hit_selection": {
        "max_hits": 100,
        "diversity_threshold": 0.7,
        "cluster_method": "butina",
        "cluster_threshold": 0.6
    },

    "docking": {
        "max_molecules": 100,  # Максимальное количество молекул для реального докинга
        "use_approximation": True,  # Использовать приближение для остальных молекул
        "timeout_per_ligand": 300,  # Таймаут в секундах для каждого лиганда
    }
}

# --- Logging и мониторинг ---
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
    "file_logging": True,
    "console_logging": True
}

# --- Параметры для экспериментов ---
EXPERIMENT_CONFIG = {
    "track_experiments": True,
    "experiment_name": f"DYRK1A_Alzheimer_Discovery_{CHOSEN_TARGET_ID}",
    "description": "Drug discovery pipeline for DYRK1A inhibitors targeting Alzheimer's disease",
    "tags": ["DYRK1A", "Alzheimer", "neurodegeneration", "kinase_inhibitor"],
    "save_models": True,
    "save_results": True,
    "generate_reports": True
}
