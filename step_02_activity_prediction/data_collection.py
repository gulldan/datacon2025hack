from pathlib import Path

import polars as pl
from chembl_webresource_client.new_client import new_client

import config
from utils.logger import LOGGER

ALLOWED_TYPES: list[str] = ["IC50", "Ki", "EC50"]
ALLOWED_UNITS: list[str] = ["nm", "um", "µm", "μm"]  # разные варианты микромолей


def _convert_to_nm(value: float, units: str) -> float | None:
    """Конвертирует значение активности в нМ.

    Args:
        value (float): Значение активности.
        units (str): Единицы измерения (nm/µM/μM/uM/UM).

    Returns:
        float | None: Значение в нМ или ``None``, если конвертация невозможна.
    """
    if value is None:
        return None

    units = units.lower().replace("μ", "u").replace("µ", "u")  # нормализуем микросимвол
    if units == "nm":
        return float(value)
    if units == "um":
        return float(value) * 1_000.0  # 1 µМ = 1000 нМ
    # Если другие единицы, пропускаем
    return None


def download_raw_data(target_id: str, output_path: Path) -> None:
    """Скачивает все активности для указанной мишени из ChEMBL и сохраняет в Parquet.

    Если файл уже существует, повторная загрузка не выполняется.
    """
    if output_path.exists():
        LOGGER.info(f"Raw data file {output_path} already exists – skipping download.")
        return

    LOGGER.info(f"Downloading ChEMBL activities for target {target_id}…")
    activity = new_client.activity  # type: ignore[attr-defined]

    # Формируем фильтр: все записи по мишени
    res = list(activity.filter(target_chembl_id=target_id))

    # Оставляем только необходимые поля и приводим их к единой схеме
    key_cols = [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_units",
        "standard_type",
    ]

    cleaned_records = []
    for entry in res:
        std_type = entry.get("standard_type")
        if std_type not in ALLOWED_TYPES:
            continue  # пропускаем типы, которые нас не интересуют

        record = {k: entry.get(k) for k in key_cols}
        cleaned_records.append(record)

    df_raw = pl.from_dicts(cleaned_records)
    df_raw.write_parquet(output_path)
    LOGGER.info(f"Saved raw activities to {output_path} with {len(df_raw)} records.")


def clean_dataset(input_path: Path, output_path: Path) -> pl.DataFrame:
    """Очищает сырые данные и возвращает обработанный DataFrame.

    Шаги:
        1. Фильтрация по типу активности, наличию значений и SMILES.
        2. Конвертация значений в нМ.
        3. Удаление дубликатов по SMILES.
        4. Удаление выбросов по IQR (по колонки value_nM).
    """
    LOGGER.info("Cleaning raw activity data…")
    df = pl.read_parquet(input_path)

    # 1. Базовая фильтрация
    df = df.filter(
        (pl.col("standard_type").is_in(ALLOWED_TYPES))
        & (pl.col("standard_value").is_not_null())
        & (pl.col("canonical_smiles").is_not_null())
        & (pl.col("standard_units").is_not_null())
    )

    # 2. Нормализация единиц и конвертация
    df = df.with_columns(pl.col("standard_units").str.to_lowercase().alias("_units_norm"))

    # Вычисляем value_nM в чистом Python для совместимости с разными версиями Polars
    value_nm_list = [
        _convert_to_nm(val, unit)  # type: ignore[arg-type]
        for val, unit in zip(df["standard_value"].to_list(), df["_units_norm"].to_list(), strict=False)
    ]
    df = df.with_columns(pl.Series(name="value_nM", values=value_nm_list))
    df = df.filter(pl.col("value_nM").is_not_null())

    # 3. Удаляем дубликаты по SMILES (оставляем запись с наименьшим значением IC50)
    df = df.sort("value_nM").unique(subset=["canonical_smiles"], keep="first")

    # 4. Удаление выбросов (IQR)
    q1, q3 = df.select(pl.col("value_nM").quantile(0.25).alias("q1"), pl.col("value_nM").quantile(0.75).alias("q3")).row(0)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df = df.filter((pl.col("value_nM") >= lower) & (pl.col("value_nM") <= upper))

    # 5. Финальные колонки
    df = df.select(
        molecule_chembl_id="molecule_chembl_id",
        SMILES="canonical_smiles",
        IC50_nM="value_nM",
        standard_type="standard_type",
        standard_units="_units_norm",
    )

    df.write_parquet(output_path)
    LOGGER.info(f"Saved cleaned dataset to {output_path} with {len(df)} molecules.")
    return df


def main():
    LOGGER.info("=== Step 1: Data collection and preprocessing ===")
    download_raw_data(config.CHOSEN_TARGET_ID, config.ACTIVITY_DATA_RAW_PATH)
    clean_dataset(config.ACTIVITY_DATA_RAW_PATH, config.ACTIVITY_DATA_PROCESSED_PATH)
    LOGGER.info("=== Data collection finished ===")


if __name__ == "__main__":
    main()
