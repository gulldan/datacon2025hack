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

# Выбранная мишень (Пример: GSK-3 beta)
# CHEMBL ID для GSK3B человека: CHEMBL279
# PDB ID для структуры с лигандом: например, 1Q41
CHOSEN_TARGET_ID = "CHEMBL279"
CHOSEN_PDB_ID = "1Q41"

# --- Шаг 2: Предсказание активности ---
PREDICTION_DIR = BASE_DIR / "step_02_activity_prediction"
PREDICTION_RESULTS_DIR = PREDICTION_DIR / "results"
PREDICTION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITY_DATA_RAW_PATH = RAW_DATA_DIR / f"{CHOSEN_TARGET_ID}_raw.parquet"
ACTIVITY_DATA_PROCESSED_PATH = PROCESSED_DATA_DIR / f"{CHOSEN_TARGET_ID}_processed.parquet"
MODEL_PATH = PREDICTION_RESULTS_DIR / "activity_model.joblib"
EDA_PLOTS_PATH = PREDICTION_RESULTS_DIR / "eda_plots.html"
FEATURE_IMPORTANCE_PATH = PREDICTION_RESULTS_DIR / "feature_importance.html"

# --- Шаг 3: Генерация молекул ---
GENERATION_DIR = BASE_DIR / "step_03_molecule_generation"
GENERATION_RESULTS_DIR = GENERATION_DIR / "results"
GENERATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GENERATED_MOLECULES_PATH = GENERATION_RESULTS_DIR / "generated_molecules.parquet"

# --- Шаг 4: Отбор хитов ---
HIT_SELECTION_DIR = BASE_DIR / "step_04_hit_selection"
HIT_SELECTION_RESULTS_DIR = HIT_SELECTION_DIR / "results"
HIT_SELECTION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_HITS_PATH = HIT_SELECTION_RESULTS_DIR / "final_hits.parquet"
DOCKING_RESULTS_PATH = HIT_SELECTION_RESULTS_DIR / "docking_scores.parquet"

# --- Параметры моделей ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
