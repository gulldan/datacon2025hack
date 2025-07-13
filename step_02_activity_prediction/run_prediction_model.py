# step_02_activity_prediction/run_prediction_model.py
from pathlib import Path

import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from chembl_webresource_client.new_client import new_client
from plotly.subplots import make_subplots
from polars_ds.linear_models import ElasticNet
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

import config
from utils.logger import LOGGER

# --- Функции ---

def download_chembl_data(target_id: str, output_path: Path):
    """Загружает данные по активности для заданной мишени из ChEMBL.

    Args:
        target_id (str): ID мишени в ChEMBL.
        output_path (Path): Путь для сохранения сырых данных.
    """
    if output_path.exists():
        LOGGER.info(f"Файл с данными {output_path} уже существует. Загрузка пропускается.")
        return

    LOGGER.info(f"Загрузка данных для мишени {target_id} из ChEMBL...")
    activity = new_client.activity
    res = activity.filter(target_chembl_id=target_id).filter(standard_type="IC50")

    df = pl.DataFrame(res)
    df.write_parquet(output_path)
    LOGGER.info(f"Данные сохранены в {output_path}. Количество записей: {len(df)}")

def process_data(input_path: Path, output_path: Path) -> pl.DataFrame:
    """Обрабатывает сырые данные: фильтрация, очистка, вычисление pIC50.

    Args:
        input_path (Path): Путь к сырым данным.
        output_path (Path): Путь для сохранения обработанных данных.

    Returns:
        pl.DataFrame: Обработанный датафрейм.
    """
    LOGGER.info("Обработка данных...")
    df = pl.read_parquet(input_path)

    # 1. Фильтрация
    df = df.filter(
        (pl.col("standard_value").is_not_null()) &
        (pl.col("canonical_smiles").is_not_null()) &
        (pl.col("standard_units") == "nM")
    )

    # 2. Удаление дубликатов
    df = df.unique(subset=["canonical_smiles"], keep="first")

    # 3. Вычисление pIC50
    # pIC50 = -log10(IC50 в Молях). IC50 у нас в нМ, поэтому IC50 * 10^-9
    df = df.with_columns(
        pl.col("standard_value").cast(pl.Float64)
    ).with_columns(
        pIC50=(-pl.col("standard_value") * 1e-9).log10()
    )

    # 4. Выбор нужных колонок
    df = df.select(["molecule_chembl_id", "canonical_smiles", "pIC50"])

    df.write_parquet(output_path)
    LOGGER.info(f"Данные обработаны и сохранены в {output_path}. Итоговое количество молекул: {len(df)}")
    return df

def calculate_descriptors(smiles: str):
    """Вычисляет дескрипторы RDKit для одной молекулы.

    Args:
        smiles (str): SMILES строка молекулы.

    Returns:
        dict or None: Словарь с дескрипторами или None, если молекула невалидна.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
    }

def generate_fingerprints(smiles: str, n_bits: int = 2048):
    """Генерирует Morgan Fingerprints.

    Args:
        smiles (str): SMILES строка.
        n_bits (int): Размерность фингерпринта.

    Returns:
        np.array or None: Массив фингерпринта.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    return np.array(fp)

def run_activity_prediction_pipeline():
    """Основная функция для запуска пайплайна предсказания активности."""
    LOGGER.info("--- Запуск этапа 2: Предсказание активности ---")

    # 1. Сбор данных
    download_chembl_data(config.CHOSEN_TARGET_ID, config.ACTIVITY_DATA_RAW_PATH)

    # 2. Обработка и EDA
    df = process_data(config.ACTIVITY_DATA_RAW_PATH, config.ACTIVITY_DATA_PROCESSED_PATH)

    # EDA
    fig_eda = make_subplots(rows=1, cols=2, subplot_titles=("Распределение pIC50", "Распределение MolWt"))
    fig_eda.add_trace(go.Histogram(x=df["pIC50"], name="pIC50"), row=1, col=1)

    # Добавим дескрипторы для EDA
    descriptors = pl.DataFrame([calculate_descriptors(s) for s in df["canonical_smiles"] if s])
    fig_eda.add_trace(go.Histogram(x=descriptors["MolWt"], name="MolWt"), row=1, col=2)
    fig_eda.update_layout(title_text="Exploratory Data Analysis (EDA)")
    fig_eda.write_html(config.EDA_PLOTS_PATH)
    LOGGER.info(f"Графики EDA сохранены в {config.EDA_PLOTS_PATH}")

    # 3. Генерация признаков (дескрипторов и фингерпринтов)
    LOGGER.info("Генерация фингерпринтов...")
    X = np.array([fp for fp in df["canonical_smiles"].apply(generate_fingerprints) if fp is not None])
    y = df.filter(pl.col("canonical_smiles").is_in(df["canonical_smiles"]))["pIC50"].to_numpy()

    # 4. Обучение модели (ElasticNet)
    LOGGER.info("Обучение модели предсказания активности с помощью ElasticNet (polars_ds)...")

    # Простое разделение на train/test без sklearn
    rng = np.random.default_rng(config.RANDOM_STATE)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    test_size = int(len(X) * config.TEST_SIZE)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    model = ElasticNet(l1_reg=0.001, l2_reg=0.01, fit_bias=True, max_iter=5000)
    model.fit(X_train, y_train)

    # 5. Оценка модели
    def _rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def _r2(a: np.ndarray, b: np.ndarray) -> float:
        ss_res = float(np.sum((a - b) ** 2))
        ss_tot = float(np.sum((a - np.mean(a)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    y_pred_train = model.predict(X_train).flatten()
    y_pred_test = model.predict(X_test).flatten()

    r2_train = _r2(y_train, y_pred_train)
    r2_test = _r2(y_test, y_pred_test)
    rmse_train = _rmse(y_train, y_pred_train)
    rmse_test = _rmse(y_test, y_pred_test)

    LOGGER.info(f"Метрики на обучающей выборке: R^2 = {r2_train:.3f}, RMSE = {rmse_train:.3f}")
    LOGGER.info(f"Метрики на тестовой выборке: R^2 = {r2_test:.3f}, RMSE = {rmse_test:.3f}")

    # Сохранение модели
    joblib.dump(model, config.MODEL_PATH)
    LOGGER.info(f"Модель сохранена в {config.MODEL_PATH}")

    # 6. Интерпретация результатов
    LOGGER.info("Интерпретация результатов модели...")
    # Для ElasticNet используем абсолютные значения коэффициентов как меру важности признака
    importances = np.abs(model.coeffs())
    feature_names = [f"Bit_{i}" for i in range(X.shape[1])]

    feature_importance_df = pl.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort("importance", descending=True).head(20)

    fig_importance = px.bar(
        feature_importance_df.to_pandas(),
        x="importance",
        y="feature",
        orientation="h",
        title="Топ-20 Важных признаков (битов фингерпринта)"
    )
    fig_importance.update_layout(yaxis={"categoryorder":"total ascending"})
    fig_importance.write_html(config.FEATURE_IMPORTANCE_PATH)
    LOGGER.info(f"График важности признаков сохранен в {config.FEATURE_IMPORTANCE_PATH}")
    LOGGER.info("""
    Интерпретация важности признаков:
    - График показывает, какие биты в Morgan Fingerprint вносят наибольший вклад в предсказание pIC50.
    - Каждый бит представляет наличие определенного химического подструктурного окружения.
    - Для глубокого анализа можно было бы использовать SHAP или сопоставить самые важные биты с конкретными химическими фрагментами в наиболее активных молекулах. Это помогло бы понять, какие функциональные группы отвечают за рост или снижение активности.
    """)
    LOGGER.info("--- Этап 2 завершен ---")

if __name__ == "__main__":
    run_activity_prediction_pipeline()
