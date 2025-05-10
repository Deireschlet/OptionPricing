import functools
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Callable, Any


def get_logger(name: str = "NO_NAME") -> logging.Logger:
    """
    Returns a logger with the given name.

    :param name: Name of the logger.
    :return: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # Standard formatter
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler with level INFO
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Rotating file handler with level DEBUG
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"{name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,             # Keep 3 backups
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_call(log: logging.Logger):
    """
    Creates a decorator that logs a function call.
    :param log: accepts logger as an argument
    :return: decorator
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log.debug(f"Called function: {func.__name__} - Args: {args} - Kwarg: {kwargs}")
            return func(*args, **kwargs)
        return wrapper
    return decorator
