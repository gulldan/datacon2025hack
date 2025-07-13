# utils/logger.py
import sys

from loguru import logger


def setup_logger():
    """Настраивает и возвращает кастомный логгер."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    return logger


LOGGER = setup_logger()
