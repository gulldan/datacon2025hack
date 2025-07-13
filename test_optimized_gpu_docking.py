#!/usr/bin/env python3
"""Optimized GPU Docking Test Script
Tests the improved GPU utilization with monitoring and performance analysis
"""

import sys
import time
from pathlib import Path

import polars as pl

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import DOCKING_PARAMETERS
from gpu_monitoring import GPUMonitor
from step_04_hit_selection.accelerated_docking import AcceleratedDocking
from utils.logger import setup_logger

logger = setup_logger()

class OptimizedGPUDockingTest:
    """Тест оптимизированного GPU докинга с мониторингом производительности"""

    def __init__(self):
        self.monitor = GPUMonitor(log_file="optimized_gpu_test.log", interval=0.5)
        self.docking = AcceleratedDocking(DOCKING_PARAMETERS)
        self.results = []

    def prepare_test_molecules(self, count: int = 500) -> list[dict]:
        """Подготавливает тестовые молекулы для докинга"""
        logger.info(f"Подготовка {count} тестовых молекул...")

        # Загружаем отфильтрованные молекулы
        try:
            molecules_df = pl.read_parquet("step_03_molecule_generation/results/filtered_molecules.parquet")

            # Берем первые N молекул
            test_molecules = molecules_df.head(count).to_dicts()

            logger.info(f"Загружено {len(test_molecules)} молекул для тестирования")
            return test_molecules

        except Exception as e:
            logger.error(f"Ошибка загрузки молекул: {e}")

            # Создаем синтетические тестовые молекулы
            test_molecules = []
            for i in range(count):
                test_molecules.append({
                    "smiles": f"C1=CC=C(C=C1)C(=O)N{i}",  # Простая структура
                    "score": 0.5,
                    "molecule_id": f"test_mol_{i}"
                })

            logger.info(f"Создано {len(test_molecules)} синтетических молекул")
            return test_molecules

    def run_performance_test(self, batch_sizes: list[int] = [100, 500, 1000, 2000]):
        """Запускает тест производительности с различными размерами батчей"""
        logger.info("🚀 Запуск теста производительности GPU докинга")

        results = []

        for batch_size in batch_sizes:
            logger.info(f"\n📊 Тестирование с размером батча: {batch_size}")

            # Подготавливаем молекулы
            test_molecules = self.prepare_test_molecules(batch_size)

            # Обновляем конфигурацию
            self.docking.config["batch_size"] = batch_size
            self.docking.batch_size = batch_size

            # Запускаем мониторинг
            self.monitor.start_monitoring()

            # Запускаем докинг
            start_time = time.time()

            try:
                scores = self.docking.dock_molecules_batch(test_molecules)
                end_time = time.time()

                # Останавливаем мониторинг
                self.monitor.stop_monitoring()

                # Получаем статистику производительности
                performance_stats = self.monitor.get_performance_summary()

                # Анализируем результаты
                total_time = end_time - start_time
                successful_dockings = len([s for s in scores.values() if s is not None and s < 0])
                throughput = len(scores) / total_time if total_time > 0 else 0

                batch_results = {
                    "batch_size": batch_size,
                    "total_molecules": len(test_molecules),
                    "successful_dockings": successful_dockings,
                    "total_time": total_time,
                    "throughput": throughput,
                    "gpu_utilization_avg": performance_stats.get("gpu_utilization", {}).get("avg", 0),
                    "gpu_utilization_max": performance_stats.get("gpu_utilization", {}).get("max", 0),
                    "cpu_utilization_avg": performance_stats.get("cpu_utilization", {}).get("avg", 0),
                    "best_score": min(scores.values()) if scores else None,
                    "scores_count": len(scores)
                }

                results.append(batch_results)

                # Выводим результаты
                logger.info(f"✅ Батч {batch_size} завершен:")
                logger.info(f"   Время: {total_time:.2f}s")
                logger.info(f"   Производительность: {throughput:.2f} молекул/сек")
                logger.info(f"   Успешных докингов: {successful_dockings}/{len(test_molecules)}")
                logger.info(f"   GPU утилизация: {batch_results['gpu_utilization_avg']:.1f}% (avg), {batch_results['gpu_utilization_max']:.1f}% (max)")
                logger.info(f"   CPU утилизация: {batch_results['cpu_utilization_avg']:.1f}% (avg)")

                if batch_results["best_score"]:
                    logger.info(f"   Лучший скор: {batch_results['best_score']:.2f}")

            except Exception as e:
                logger.error(f"❌ Ошибка в тестировании батча {batch_size}: {e}")
                self.monitor.stop_monitoring()

                results.append({
                    "batch_size": batch_size,
                    "error": str(e),
                    "total_time": 0,
                    "throughput": 0,
                    "gpu_utilization_avg": 0
                })

            # Пауза между тестами
            time.sleep(5)

        return results

    def run_stress_test(self, duration_minutes: int = 10):
        """Запускает стресс-тест GPU на заданное время"""
        logger.info(f"🔥 Запуск стресс-теста GPU на {duration_minutes} минут")

        # Подготавливаем большой набор молекул
        test_molecules = self.prepare_test_molecules(5000)

        # Настраиваем для максимальной нагрузки
        self.docking.config["batch_size"] = 2000
        self.docking.config["max_concurrent_jobs"] = 32
        self.docking.batch_size = 2000

        # Запускаем мониторинг
        self.monitor.start_monitoring()

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        total_processed = 0
        total_successful = 0

        try:
            while time.time() < end_time:
                # Запускаем докинг
                scores = self.docking.dock_molecules_batch(test_molecules)

                total_processed += len(test_molecules)
                total_successful += len([s for s in scores.values() if s is not None and s < 0])

                logger.info(f"🔄 Обработано: {total_processed}, успешно: {total_successful}")

                # Короткая пауза
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("🛑 Стресс-тест прерван пользователем")
        except Exception as e:
            logger.error(f"❌ Ошибка в стресс-тесте: {e}")
        finally:
            self.monitor.stop_monitoring()

            # Получаем итоговую статистику
            performance_stats = self.monitor.get_performance_summary()
            actual_duration = time.time() - start_time

            logger.info("\n📊 Результаты стресс-теста:")
            logger.info(f"   Длительность: {actual_duration/60:.1f} минут")
            logger.info(f"   Обработано молекул: {total_processed}")
            logger.info(f"   Успешных докингов: {total_successful}")
            logger.info(f"   Средняя производительность: {total_processed/actual_duration:.2f} молекул/сек")

            if "error" not in performance_stats:
                logger.info(f"   GPU утилизация: {performance_stats['gpu_utilization']['avg']:.1f}% (avg), {performance_stats['gpu_utilization']['max']:.1f}% (max)")
                logger.info(f"   CPU утилизация: {performance_stats['cpu_utilization']['avg']:.1f}% (avg)")

    def analyze_optimization_effectiveness(self, results: list[dict]):
        """Анализирует эффективность оптимизации"""
        logger.info("\n📈 Анализ эффективности оптимизации:")

        if not results:
            logger.warning("Нет результатов для анализа")
            return

        # Находим оптимальный размер батча
        best_result = max(results, key=lambda x: x.get("gpu_utilization_avg", 0))

        logger.info("🏆 Лучший результат:")
        logger.info(f"   Размер батча: {best_result['batch_size']}")
        logger.info(f"   GPU утилизация: {best_result['gpu_utilization_avg']:.1f}%")
        logger.info(f"   Производительность: {best_result['throughput']:.2f} молекул/сек")

        # Анализируем тренды
        logger.info("\n📊 Тренды производительности:")
        for result in results:
            if "error" not in result:
                logger.info(f"   Батч {result['batch_size']:4d}: "
                          f"GPU {result['gpu_utilization_avg']:5.1f}%, "
                          f"throughput {result['throughput']:6.2f} mol/s")

        # Рекомендации
        logger.info("\n💡 Рекомендации:")

        if best_result["gpu_utilization_avg"] < 50:
            logger.info("   ⚠️  GPU утилизация низкая - рассмотрите увеличение размера батча")
        elif best_result["gpu_utilization_avg"] > 90:
            logger.info("   🔥 Отличная GPU утилизация - система работает эффективно")
        else:
            logger.info("   ✅ Хорошая GPU утилизация - система сбалансирована")

        if best_result["throughput"] > 10:
            logger.info("   🚀 Высокая производительность - оптимизация успешна")
        else:
            logger.info("   ⚠️  Производительность можно улучшить")

def main():
    """Основная функция тестирования"""
    print("🧪 Тест оптимизированного GPU докинга")
    print("=" * 50)

    tester = OptimizedGPUDockingTest()

    try:
        # Тест производительности с разными размерами батчей
        results = tester.run_performance_test([100, 500, 1000, 2000])

        # Анализ результатов
        tester.analyze_optimization_effectiveness(results)

        # Опционально: стресс-тест
        user_input = input("\n🔥 Запустить стресс-тест на 5 минут? (y/N): ")
        if user_input.lower() == "y":
            tester.run_stress_test(duration_minutes=5)

    except KeyboardInterrupt:
        print("\n🛑 Тест прерван пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
    finally:
        print("\n✅ Тестирование завершено")

if __name__ == "__main__":
    main()
