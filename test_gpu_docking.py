#!/usr/bin/env python3
"""Тестовый скрипт для проверки GPU докинга

Этот скрипт проверяет работу GPU-ускоренного молекулярного докинга
"""

import subprocess
import sys
import time
from pathlib import Path

# Добавляем путь к корневой директории проекта
sys.path.append(str(Path(__file__).parent))

from config import DOCKING_PARAMETERS
from step_04_hit_selection.accelerated_docking import AcceleratedDocking
from step_04_hit_selection.run_hit_selection import optimize_docking_performance
from utils.logger import LOGGER as logger


def test_gpu_availability():
    """Проверка доступности GPU"""
    logger.info("=== Проверка доступности GPU ===")

    # Проверка NVIDIA GPU
    try:
        result = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✓ NVIDIA GPU доступен")
            print(result.stdout)
        else:
            logger.warning("✗ NVIDIA GPU недоступен")
    except FileNotFoundError:
        logger.warning("✗ nvidia-smi не найден")

    # Проверка AutoDock-GPU
    autodock_gpu_path = DOCKING_PARAMETERS.get("autodock_gpu_path")
    if autodock_gpu_path and Path(autodock_gpu_path).exists():
        logger.info(f"✓ AutoDock-GPU найден: {autodock_gpu_path}")

        # Тест AutoDock-GPU
        try:
            result = subprocess.run([autodock_gpu_path, "--help"], check=False, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("✓ AutoDock-GPU работает")
            else:
                logger.warning("✗ AutoDock-GPU не работает")
        except Exception as e:
            logger.warning(f"✗ Ошибка тестирования AutoDock-GPU: {e}")
    else:
        logger.warning("✗ AutoDock-GPU не найден")

    # Проверка конфигурации
    logger.info(f"GPU engine: {DOCKING_PARAMETERS.get('gpu_engine', 'не задан')}")
    logger.info(f"Use GPU: {DOCKING_PARAMETERS.get('use_gpu', False)}")


def test_docking_engine():
    """Тест докинг движка"""
    logger.info("\n=== Тест докинг движка ===")

    try:
        # Оптимизируем конфигурацию
        optimized_config = optimize_docking_performance()
        logger.info("✓ Конфигурация оптимизирована")

        # Создаем движок
        docking_engine = AcceleratedDocking(optimized_config)
        logger.info("✓ Движок создан")

        # Проверяем инициализацию
        if docking_engine.use_gpu:
            logger.info("✓ GPU режим активирован")
        else:
            logger.info("! CPU режим (GPU недоступен)")

        return docking_engine

    except Exception as e:
        logger.error(f"✗ Ошибка создания движка: {e}")
        return None


def test_simple_docking():
    """Простой тест докинга"""
    logger.info("\n=== Простой тест докинга ===")

    # Создаем движок
    docking_engine = test_docking_engine()
    if not docking_engine:
        return None

    # Тестовые молекулы
    test_molecules = [{"id": "ethanol", "smiles": "CCO"}, {"id": "methanol", "smiles": "CO"}, {"id": "water", "smiles": "O"}]

    logger.info(f"Тестируем {len(test_molecules)} молекул")

    try:
        start_time = time.time()

        # Запускаем докинг
        results = docking_engine.dock_molecules_batch(test_molecules)

        end_time = time.time()
        elapsed = end_time - start_time

        logger.info(f"✓ Докинг завершен за {elapsed:.2f} секунд")
        logger.info(f"✓ Получено {len(results)} результатов")

        # Выводим результаты
        for mol_id, score in results.items():
            logger.info(f"  {mol_id}: {score:.3f}")

        return results

    except Exception as e:
        logger.error(f"✗ Ошибка докинга: {e}")
        return None


def test_gpu_performance():
    """Тест производительности GPU"""
    logger.info("\n=== Тест производительности GPU ===")

    # Создаем движки для сравнения
    gpu_config = DOCKING_PARAMETERS.copy()
    gpu_config["use_gpu"] = True
    gpu_config["gpu_engine"] = "autodock_gpu"

    cpu_config = DOCKING_PARAMETERS.copy()
    cpu_config["use_gpu"] = False

    # Тестовые молекулы
    test_molecules = [{"id": f"test_mol_{i}", "smiles": "CCO"} for i in range(1000)]

    gpu_time = None
    cpu_time = None

    try:
        # Тест GPU
        if gpu_config["use_gpu"]:
            logger.info("Тест GPU режима...")
            gpu_engine = AcceleratedDocking(gpu_config)

            start_time = time.time()
            gpu_results = gpu_engine.dock_molecules_batch(test_molecules)
            gpu_time = time.time() - start_time

            logger.info(f"GPU: {len(gpu_results)} молекул за {gpu_time:.2f} сек")

        # Тест CPU
        logger.info("Тест CPU режима...")
        cpu_engine = AcceleratedDocking(cpu_config)

        start_time = time.time()
        cpu_results = cpu_engine.dock_molecules_batch(test_molecules)
        cpu_time = time.time() - start_time

        logger.info(f"CPU: {len(cpu_results)} молекул за {cpu_time:.2f} сек")

        # Сравнение
        if gpu_time is not None and cpu_time is not None and gpu_time > 0:
            speedup = cpu_time / gpu_time
            logger.info(f"Ускорение GPU: {speedup:.2f}x")

    except Exception as e:
        logger.error(f"✗ Ошибка тестирования производительности: {e}")


def main():
    """Основная функция тестирования"""
    logger.info("🚀 Запуск тестов GPU докинга")

    # Проверка доступности GPU
    test_gpu_availability()

    # Тест движка
    test_docking_engine()

    # Простой тест
    test_simple_docking()

    # Тест производительности
    test_gpu_performance()

    logger.info("\n✅ Тестирование завершено")


if __name__ == "__main__":
    main()
