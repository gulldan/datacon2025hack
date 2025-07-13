# step_02_activity_prediction/run_prediction_model.py
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from chembl_webresource_client.new_client import new_client
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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

    # 4. Обучение модели
    LOGGER.info("Обучение модели предсказания активности...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    model = RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    # 5. Оценка модели
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    LOGGER.info(f"Метрики на обучающей выборке: R^2 = {r2_train:.3f}, RMSE = {rmse_train:.3f}")
    LOGGER.info(f"Метрики на тестовой выборке: R^2 = {r2_test:.3f}, RMSE = {rmse_test:.3f}")

    # Сохранение модели
    joblib.dump(model, config.MODEL_PATH)
    LOGGER.info(f"Модель сохранена в {config.MODEL_PATH}")

    # 6. Интерпретация результатов
    LOGGER.info("Интерпретация результатов модели...")
    importances = model.feature_importances_
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
