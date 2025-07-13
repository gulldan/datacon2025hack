# main.py
import argparse

from utils.logger import LOGGER


def main():
    parser = argparse.ArgumentParser(description="In silico пайплайн для поиска лекарств от болезни Альцгеймера.")
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="Номер этапа для запуска (1: Target, 2: Predict, 3: Generate, 4: Select).",
    )

    args = parser.parse_args()

    if args.step == 1:
        from step_01_target_selection.run_target_analysis import generate_target_selection_report

        LOGGER.info("Запуск этапа 1: Выбор мишени")
        generate_target_selection_report()

    elif args.step == 2:
        from step_02_activity_prediction.run_prediction_model import run_activity_prediction_pipeline

        LOGGER.info("Запуск этапа 2: Предсказание активности")
        run_activity_prediction_pipeline()

    elif args.step == 3:
        from step_03_molecule_generation.run_generation import run_generation_pipeline

        LOGGER.info("Запуск этапа 3: Генерация молекул")
        run_generation_pipeline()

    elif args.step == 4:
        from step_04_hit_selection.run_hit_selection import run_hit_selection_pipeline

        LOGGER.info("Запуск этапа 4: Отбор хитов")
        run_hit_selection_pipeline()


if __name__ == "__main__":
    main()
