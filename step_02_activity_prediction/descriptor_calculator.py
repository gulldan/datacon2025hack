"""Модуль для расчёта молекулярных дескрипторов.

Этот модуль содержит функции для расчёта дескрипторов RDKit и Mordred,
а также фингерпринтов различных типов.
"""

import hashlib
import json
import logging
import tempfile
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from rdkit import Chem

# Химические библиотеки
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.AtomPairs import Pairs, Torsions

# Проверяем доступность Mordred
try:
    from mordred import Calculator, descriptors

    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False
    Calculator = None
    descriptors = None

# Проверяем доступность PaDELPy
try:
    from padelpy import padeldescriptor

    PADELPY_AVAILABLE = True
except ImportError:
    PADELPY_AVAILABLE = False
    padeldescriptor = None

logger = logging.getLogger(__name__)

# Настройка путей
DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"
DESCRIPTORS_DIR = DATA_DIR / "descriptors"

# Создаём папки если их нет
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DESCRIPTORS_DIR.mkdir(parents=True, exist_ok=True)

# Файл метаданных кэша
CACHE_METADATA_FILE = CACHE_DIR / "cache_metadata.json"


def get_optimal_threads() -> int:
    """Определяет оптимальное количество потоков для расчётов.

    Returns:
        Количество потоков для использования.
    """
    try:
        # Используем 75% от общего количества CPU cores
        threads = max(1, int(cpu_count() * 0.75))
        logger.info(f"Используется {threads} потоков (из {cpu_count()} доступных)")
        return threads
    except Exception:
        logger.warning("Не удалось определить количество CPU cores, используется 1 поток")
        return 1


def get_smiles_hash(smiles_list: list[str]) -> str:
    """Создаёт уникальный хеш для списка SMILES.

    Args:
        smiles_list: Список SMILES строк.

    Returns:
        SHA256 хеш для использования в качестве ключа кэша.
    """
    # Сортируем SMILES для создания consistent hash
    sorted_smiles = sorted(smiles_list)
    smiles_str = "\n".join(sorted_smiles)
    return hashlib.sha256(smiles_str.encode()).hexdigest()


def load_cache_metadata() -> dict:
    """Загружает метаданные кэша.

    Returns:
        Словарь с метаданными кэша.
    """
    try:
        if CACHE_METADATA_FILE.exists():
            with open(CACHE_METADATA_FILE) as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Не удалось загрузить метаданные кэша: {e}")
    return {}


def save_cache_metadata(metadata: dict) -> None:
    """Сохраняет метаданные кэша.

    Args:
        metadata: Словарь с метаданными кэша.
    """
    try:
        with open(CACHE_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.warning(f"Предупреждение: Не удалось сохранить метаданные кэша: {e}")


def get_cached_descriptors(cache_key: str, descriptor_type: str) -> pl.DataFrame | None:
    """Загружает дескрипторы из кэша.

    Args:
        cache_key: Ключ кэша.
        descriptor_type: Тип дескрипторов (rdkit, mordred, padel).

    Returns:
        DataFrame с дескрипторами или None, если кэш не найден.
    """
    cache_file = CACHE_DIR / f"{descriptor_type}_{cache_key}.parquet"

    if cache_file.exists():
        try:
            df = pl.read_parquet(cache_file)
            logger.info(f"Загружены {descriptor_type} дескрипторы из кэша ({df.shape[1]} дескрипторов)")
            return df
        except Exception as e:
            logger.warning(f"Ошибка при загрузке кэша {descriptor_type}: {e}")
            # Удаляем поврежденный кэш
            cache_file.unlink(missing_ok=True)

    return None


def save_descriptors_to_cache(df: pl.DataFrame, cache_key: str, descriptor_type: str) -> None:
    """Сохраняет дескрипторы в кэш.

    Args:
        df: DataFrame с дескрипторами.
        cache_key: Ключ кэша.
        descriptor_type: Тип дескрипторов.
    """
    cache_file = CACHE_DIR / f"{descriptor_type}_{cache_key}.parquet"

    try:
        df.write_parquet(cache_file)

        # Обновляем метаданные
        metadata = load_cache_metadata()
        metadata[f"{descriptor_type}_{cache_key}"] = {
            "type": descriptor_type,
            "shape": df.shape,
            "created": pd.Timestamp.now().isoformat(),
            "file": str(cache_file),
        }
        save_cache_metadata(metadata)

        logger.info(f"Сохранены {descriptor_type} дескрипторы в кэш ({df.shape[1]} дескрипторов)")
    except Exception as e:
        logger.warning(f"Ошибка при сохранении в кэш {descriptor_type}: {e}")


def calculate_rdkit_descriptors(smiles_list: list[str], use_cache: bool = True) -> pl.DataFrame:
    """Расчёт всех доступных дескрипторов RDKit для списка SMILES.

    Args:
        smiles_list: Список SMILES строк для расчёта дескрипторов.
        use_cache: Использовать кэширование результатов.

    Returns:
        Polars DataFrame с рассчитанными дескрипторами RDKit.

    Raises:
        ValueError: Если список SMILES пуст.
    """
    if not smiles_list:
        raise ValueError("Список SMILES не может быть пустым")

    # Проверяем кэш
    cache_key = get_smiles_hash(smiles_list)
    if use_cache:
        cached_result = get_cached_descriptors(cache_key, "rdkit")
        if cached_result is not None:
            return cached_result

    # Получаем список всех дескрипторов RDKit
    descriptor_list = [x[0] for x in Descriptors._descList]
    logger.info(f"Найдено {len(descriptor_list)} дескрипторов RDKit")

    results = []
    failed_molecules = 0

    for i, smiles in enumerate(smiles_list):
        if i % 500 == 0:
            logger.info(f"Обработано {i}/{len(smiles_list)} молекул...")

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failed_molecules += 1
                results.append([None] * len(descriptor_list))
                continue

            # Рассчитываем все дескрипторы
            descriptors_values = []
            for desc_name in descriptor_list:
                try:
                    desc_fn = getattr(Descriptors, desc_name)
                    value = desc_fn(mol)
                    # Конвертируем NaN в None для Polars
                    if np.isnan(value) if isinstance(value, (int, float)) else False:
                        descriptors_values.append(None)
                    else:
                        descriptors_values.append(value)
                except Exception:
                    descriptors_values.append(None)

            results.append(descriptors_values)

        except Exception:
            failed_molecules += 1
            results.append([None] * len(descriptor_list))

    logger.info(f"Не удалось обработать {failed_molecules} молекул")

    # Создаём Polars DataFrame с дескрипторами
    # Используем explicit orient для избежания warning
    descriptors_df = pl.DataFrame(data=results, schema=descriptor_list, orient="row")

    # Сохраняем в кэш
    if use_cache:
        save_descriptors_to_cache(descriptors_df, cache_key, "rdkit")

    return descriptors_df


def calculate_mordred_descriptors(smiles_list: list[str], ignore_3d: bool = True, use_cache: bool = True) -> pl.DataFrame | None:
    """Расчёт дескрипторов Mordred для списка SMILES.

    Args:
        smiles_list: Список SMILES строк для расчёта дескрипторов.
        ignore_3d: Если True, игнорировать 3D дескрипторы.
        use_cache: Использовать кэширование результатов.

    Returns:
        Polars DataFrame с рассчитанными дескрипторами Mordred или None при ошибке.

    Raises:
        ValueError: Если список SMILES пуст.
    """
    if not MORDRED_AVAILABLE:
        logger.warning("Mordred не доступен")
        return None

    if not smiles_list:
        raise ValueError("Список SMILES не может быть пустым")

    # Проверяем кэш
    cache_key = get_smiles_hash(smiles_list)
    if use_cache:
        cached_result = get_cached_descriptors(cache_key, "mordred")
        if cached_result is not None:
            return cached_result

    # Создаём калькулятор для всех 2D дескрипторов
    calc = Calculator(descriptors, ignore_3D=True) if ignore_3d else Calculator(descriptors)

    logger.info(f"Найдено {len(calc.descriptors)} дескрипторов Mordred")

    # Конвертируем SMILES в молекулы
    molecules = []
    failed_indices = []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed_indices.append(i)
            molecules.append(None)
        else:
            molecules.append(mol)

    logger.info(f"Не удалось конвертировать {len(failed_indices)} SMILES")

    # Рассчитываем дескрипторы
    logger.info("Расчёт дескрипторов Mordred (может занять несколько минут)...")

    try:
        # Используем pandas для расчёта, затем конвертируем в polars
        results_pd = calc.pandas(molecules)

        # Более тщательная обработка Error объектов
        import pandas as pd

        def clean_error_values(x):
            """Очистка Error объектов из Mordred."""
            if pd.isna(x):
                return None
            if hasattr(x, "__class__") and ("Error" in str(type(x)) or "error" in str(type(x)).lower()):
                return None
            try:
                # Попытка конвертации в float
                if isinstance(x, (int, float)):
                    return float(x) if not pd.isna(x) else None
                return float(x) if str(x).replace(".", "").replace("-", "").isdigit() else None
            except (ValueError, TypeError):
                return None

        # Применяем очистку ко всем колонкам
        for col in results_pd.columns:
            results_pd[col] = results_pd[col].apply(clean_error_values)

        # Конвертируем pandas DataFrame в polars DataFrame
        results = pl.from_pandas(results_pd)
        logger.info(f"Успешно: Рассчитано {results.shape[1]} дескрипторов Mordred")

        # Сохраняем в кэш
        if use_cache:
            save_descriptors_to_cache(results, cache_key, "mordred")

        return results
    except Exception as e:
        logger.error(f"Ошибка: Ошибка при расчёте Mordred: {e}")
        return None


def calculate_fingerprints(smiles_list: list[str], use_cache: bool = True) -> dict[str, pl.DataFrame]:
    """Расчёт различных типов фингерпринтов для списка SMILES.

    Args:
        smiles_list: Список SMILES строк для расчёта фингерпринтов.
        use_cache: Использовать кэширование результатов.

    Returns:
        Словарь с Polars DataFrames для каждого типа фингерпринтов.

    Raises:
        ValueError: Если список SMILES пуст.
    """
    if not smiles_list:
        raise ValueError("Список SMILES не может быть пустым")

    # Проверяем кэш для каждого типа фингерпринтов
    cache_key = get_smiles_hash(smiles_list)
    fingerprint_types = ["morgan_2048", "morgan_1024", "maccs", "atompairs", "topological"]
    cached_results = {}

    if use_cache:
        for fp_type in fingerprint_types:
            cached_result = get_cached_descriptors(cache_key, f"fingerprint_{fp_type}")
            if cached_result is not None:
                cached_results[fp_type] = cached_result

    # Если все фингерпринты в кэше, возвращаем их
    if len(cached_results) == len(fingerprint_types):
        logger.info("Кэш: Все фингерпринты загружены из кэша")
        return cached_results

    morgan_gen_2048 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    morgan_gen_1024 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    fingerprints_data = {
        "morgan_2048": [],  # Morgan fingerprints (ECFP), radius=2, 2048 bits
        "morgan_1024": [],  # Morgan fingerprints, 1024 bits
        "maccs": [],  # MACCS keys (166 bits)
        "atompairs": [],  # Atom pairs fingerprints
        "topological": [],  # Topological torsions
    }

    failed_molecules = 0

    for i, smiles in enumerate(smiles_list):
        if i % 500 == 0:
            logger.info(f"Обработано {i}/{len(smiles_list)} молекул для фингерпринтов...")

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failed_molecules += 1
                # Добавляем пустые векторы для неудачных молекул
                fingerprints_data["morgan_2048"].append(np.zeros(2048, dtype=int))
                fingerprints_data["morgan_1024"].append(np.zeros(1024, dtype=int))
                fingerprints_data["maccs"].append(np.zeros(166, dtype=int))
                fingerprints_data["atompairs"].append(np.zeros(2048, dtype=int))
                fingerprints_data["topological"].append(np.zeros(2048, dtype=int))
                continue

                # Morgan fingerprints (ECFP) - используем новый API
            morgan_2048 = morgan_gen_2048.GetFingerprint(mol)
            morgan_1024 = morgan_gen_1024.GetFingerprint(mol)
            fingerprints_data["morgan_2048"].append(np.array(morgan_2048))
            fingerprints_data["morgan_1024"].append(np.array(morgan_1024))

            # MACCS keys
            maccs = rdMolDescriptors.GetMACCSKeysAsBitVect(mol)
            fingerprints_data["maccs"].append(np.array(maccs))

            # Atom pairs
            atompairs = Pairs.GetAtomPairFingerprintAsBitVect(mol, nBits=2048)
            fingerprints_data["atompairs"].append(np.array(atompairs))

            # Topological torsions
            torsions = Torsions.GetTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)
            fingerprints_data["topological"].append(np.array(torsions))

        except Exception:
            failed_molecules += 1
            # Добавляем пустые векторы для ошибочных молекул
            fingerprints_data["morgan_2048"].append(np.zeros(2048, dtype=int))
            fingerprints_data["morgan_1024"].append(np.zeros(1024, dtype=int))
            fingerprints_data["maccs"].append(np.zeros(166, dtype=int))
            fingerprints_data["atompairs"].append(np.zeros(2048, dtype=int))
            fingerprints_data["topological"].append(np.zeros(2048, dtype=int))

    logger.info(f"Предупреждение: Не удалось обработать {failed_molecules} молекул для фингерпринтов")

    # Конвертируем в Polars DataFrames для удобства
    fingerprint_dfs = {}
    for fp_type, fp_data in fingerprints_data.items():
        # Используем кэшированный результат если доступен
        if fp_type in cached_results:
            fingerprint_dfs[fp_type] = cached_results[fp_type]
            logger.info(f"Кэш: {fp_type}: загружен из кэша")
            continue

        fp_array = np.array(fp_data)
        n_bits = fp_array.shape[1]

        # Создаём имена колонок
        col_names = [f"{fp_type}_bit_{i}" for i in range(n_bits)]
        df = pl.DataFrame(fp_array, schema=col_names)
        fingerprint_dfs[fp_type] = df

        # Сохраняем в кэш
        if use_cache:
            save_descriptors_to_cache(df, cache_key, f"fingerprint_{fp_type}")

        logger.info(f"Успешно: {fp_type}: {fp_array.shape[1]} бит")

    return fingerprint_dfs


def calculate_padel_descriptors(
    smiles_list: list[str],
    fingerprints: bool = True,
    d_2d: bool = True,
    d_3d: bool = False,
    use_cache: bool = True,
    threads: int | None = None,
) -> pl.DataFrame | None:
    """Расчёт PaDEL дескрипторов для списка SMILES с оптимизацией.

    Args:
        smiles_list: Список SMILES строк для расчёта дескрипторов.
        fingerprints: Если True, рассчитывать фингерпринты.
        d_2d: Если True, рассчитывать 2D дескрипторы.
        d_3d: Если True, рассчитывать 3D дескрипторы.
        use_cache: Использовать кэширование результатов.
        threads: Количество потоков (None для автоматического определения).

    Returns:
        Polars DataFrame с рассчитанными PaDEL дескрипторами или None при ошибке.

    Raises:
        ValueError: Если список SMILES пуст.
    """
    if not PADELPY_AVAILABLE:
        logger.warning("PaDELPy не доступен")
        return None

    if not smiles_list:
        raise ValueError("Список SMILES не может быть пустым")

    # Проверяем кэш
    cache_key = get_smiles_hash(smiles_list)
    if use_cache:
        cached_result = get_cached_descriptors(cache_key, "padel")
        if cached_result is not None:
            return cached_result

    # Определяем количество потоков
    if threads is None:
        threads = get_optimal_threads()

        logger.info(f"Расчет: Расчёт PaDEL дескрипторов для {len(smiles_list)} молекул...")
    logger.info(f"Используется: Используется {threads} потоков")

    # Инициализируем переменные для очистки
    input_file = None
    output_csv = None

    try:
        # Создаём временные файлы в папке data
        temp_dir = DATA_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)

        # Создаём временный файл с SMILES
        with tempfile.NamedTemporaryFile(mode="w", suffix=".smi", delete=False, dir=temp_dir) as f:
            for i, smiles in enumerate(smiles_list):
                f.write(f"{smiles}\tmol_{i}\n")
            input_file = f.name

        # Файл для результатов
        output_csv = temp_dir / f"cox2_padel_descriptors_{cache_key[:8]}.csv"

        # Запускаем PaDEL-Descriptor с оптимизацией
        if padeldescriptor is not None:
            padeldescriptor(
                mol_dir=input_file,
                d_file=str(output_csv),
                fingerprints=fingerprints,
                d_2d=d_2d,
                d_3d=d_3d,
                removesalt=True,
                log=True,
                threads=threads,
                maxruntime=30000,  # 30 секунд на молекулу
                retainorder=True,  # Сохранять порядок
                standardizetautomers=True,  # Стандартизировать таутомеры
            )

        # Читаем результаты
        if output_csv.exists():
            # Читаем CSV с результатами
            results_df = pl.read_csv(output_csv)

            # Удаляем временные файлы
            Path(input_file).unlink(missing_ok=True)
            output_csv.unlink(missing_ok=True)

            logger.info(f"Успешно: Рассчитано {results_df.shape[1] - 1} PaDEL дескрипторов")  # -1 для Name колонки

            # Удаляем Name колонку (не нужна)
            if "Name" in results_df.columns:
                results_df = results_df.drop("Name")

            # Сохраняем в кэш
            if use_cache:
                save_descriptors_to_cache(results_df, cache_key, "padel")

            return results_df

        logger.error("Ошибка: PaDEL не создал выходной файл")
        # Очистка
        Path(input_file).unlink(missing_ok=True)
        return None

    except Exception as e:
        logger.error(f"Ошибка: Ошибка при расчёте PaDEL: {e}")
        # Очистка в случае ошибки
        try:
            if "input_file" in locals() and input_file:
                Path(input_file).unlink(missing_ok=True)
            if "output_csv" in locals() and output_csv:
                Path(output_csv).unlink(missing_ok=True)
        except Exception:
            pass
        return None


def save_descriptors_to_data(df: pl.DataFrame, filename: str, descriptor_type: str) -> None:
    """Сохраняет дескрипторы в папку /data/descriptors.

    Args:
        df: DataFrame с дескрипторами.
        filename: Имя файла без расширения.
        descriptor_type: Тип дескрипторов для логирования.
    """
    try:
        # Сохраняем в parquet и CSV
        parquet_path = DESCRIPTORS_DIR / f"{filename}.parquet"
        csv_path = DESCRIPTORS_DIR / f"{filename}.csv"

        df.write_parquet(parquet_path)
        df.write_csv(csv_path)

        logger.info(f"Сохранено: Сохранены {descriptor_type} дескрипторы:")
        logger.info(f"   Отчет: {parquet_path}")
        logger.info(f"   Отчет: {csv_path}")

    except Exception as e:
        logger.error(f"Ошибка: Ошибка при сохранении {descriptor_type} дескрипторов: {e}")


def clear_cache(descriptor_type: str | None = None) -> None:
    """Очищает кэш дескрипторов.

    Args:
        descriptor_type: Тип дескрипторов для очистки (None для всех).
    """
    try:
        if descriptor_type:
            # Очищаем кэш конкретного типа
            cache_files = list(CACHE_DIR.glob(f"{descriptor_type}_*.parquet"))
            for file in cache_files:
                file.unlink()
            logger.info(f"Очистка: Очищен кэш {descriptor_type} дескрипторов ({len(cache_files)} файлов)")
        else:
            # Очищаем весь кэш
            cache_files = list(CACHE_DIR.glob("*.parquet"))
            for file in cache_files:
                file.unlink()

            # Очищаем метаданные
            if CACHE_METADATA_FILE.exists():
                CACHE_METADATA_FILE.unlink()

            logger.info(f"Очистка: Очищен весь кэш ({len(cache_files)} файлов)")

    except Exception as e:
        logger.error(f"Ошибка: Ошибка при очистке кэша: {e}")


def get_cache_info() -> dict:
    """Возвращает информацию о кэше.

    Returns:
        Словарь с информацией о кэше.
    """
    try:
        metadata = load_cache_metadata()
        cache_files = list(CACHE_DIR.glob("*.parquet"))

        info = {
            "total_files": len(cache_files),
            "total_size_mb": sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
            "entries": metadata,
        }

        return info

    except Exception as e:
        logger.error(f"Ошибка: Ошибка при получении информации о кэше: {e}")
        return {}
