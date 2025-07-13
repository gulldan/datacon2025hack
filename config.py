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
# Выбор генеративной модели: "selfies_vae" | "graph_flow" (будущий T21)
GENERATOR_TYPE = "selfies_vae"
# Путь к сохранённой модели генерации графов (если выбран graph_flow)
GRAPH_FLOW_MODEL_PATH = GENERATION_RESULTS_DIR / "graph_flow.pt"

# --- Шаг 3: Гиперпараметры генеративной SELFIES-VAE и скоринга ---
# Основные размеры сети
VAE_EMBED_DIM = 196
VAE_HIDDEN_DIM = 392
VAE_LATENT_DIM = 128
VAE_NUM_LAYERS = 2
VAE_DROPOUT = 0.2
# Обучение
VAE_BATCH_SIZE = 64
VAE_MAX_LEN = 100             # сократим длину, чтобы снизить бессмысленные цепочки
VAE_LEARNING_RATE = 1e-3
# Увеличим patience, раз обучаем дольше
VAE_PATIENCE = 8
# Максимальное число эпох (можно переопределить переменной окружения)
MAX_VAE_EPOCHS = 150
# Выборка после обучения
VAE_GENERATE_N = 5000
# Максимальный размер батча при семплинге (GPU память)
VAE_SAMPLE_BATCH = 512

# Аугментация данных (рандомные SMILES на молекулу)
AUG_PER_MOL = 10

# Весовые коэффициенты финального скоринга генерации (сумма = 1.0)
SCORING_WEIGHTS = {
    "activity": 0.4,
    "qed": 0.2,
    "sa": 0.2,
    "bbbp": 0.2,
}

# --- ADMET фильтры ---
USE_CYP450_FILTERS = True                # применять ли CYP450 фильтр
CYP450_ISOFORMS = ["1A2", "2C9", "2C19", "2D6", "3A4"]  # ключевые изоферменты
# BRENK / токсофоры фильтр (T30)
USE_BRENK_FILTER = True                  # применять ли набор BRENK substructure filters
USE_HEPATOTOX_FILTER = True              # фильтр потенциальной гепатотоксичности

# --- Optuna автоматический подбор гиперпараметров ---
OPTUNA_TUNE_XGB = False                  # вкл/выкл поиск для XGBoost
OPTUNA_TUNE_VAE = False                  # вкл/выкл поиск для VAE/генераторов
OPTUNA_STUDIES_DIR = BASE_DIR / "optuna_studies"
OPTUNA_STUDIES_DIR.mkdir(parents=True, exist_ok=True)

# --- Docking parameters ---
# Позволяет отключить реальный запуск AutoDock Vina (например, если не установлен)
USE_VINA_DOCKING = True

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

# --- Docking box parameters (example values; adjust as needed) ---
# Center of grid box (Å)
BOX_CENTER = (16.5, 9.8, 25.7)
# Size of grid box (Å)
BOX_SIZE = (20.0, 20.0, 20.0)

# --- PaDEL Descriptor ---
# Путь к PaDEL-Descriptor.jar (скачайте с https://github.com/dataprofessor/padel)
PADEL_JAR_PATH = BASE_DIR / "external" / "PaDEL-Descriptor.jar"
USE_PADEL_DESCRIPTORS = False  # установить True, если Java и PaDEL.jar доступны

# Параметры моделей ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
