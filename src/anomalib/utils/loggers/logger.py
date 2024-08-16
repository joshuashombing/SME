import logging
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Union


class LogCode(str, Enum):
    BASE = "I"
    GUI = "GUI"
    ENGINE = "AI"
    RELAY = "R"
    CAMERA = "C"


def _get_now_str(timestamp: Union[float, None] = None, microsecond=False) -> str:
    """Standard format for datetimes is defined here."""

    if timestamp is None:
        timestamp = datetime.fromtimestamp(time.time())

    string = timestamp.strftime("%Y-%m-%d_%H-%M")

    if microsecond:
        string += f"-{timestamp.microsecond}"

    return string


_LOG_DIR = Path(__file__).parents[4] / "logs"
_FILENAME = f"{_get_now_str()}.log"


def setup_logger(name=None, level=logging.DEBUG, log_dir=None):
    """
    Sets up a logger to log messages to both a file and the terminal.

    Args:
        :param name: (str) Logger name.
        :param level: (int) The logging level (e.g., logging.DEBUG, logging.INFO)
        :param log_dir:
    """
    # Create a custom logger
    logger = logging.getLogger(name=name)

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set the logging level
    logger.setLevel(level)

    if log_dir is None:
        log_dir = _LOG_DIR

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / _FILENAME
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    # Set the logging level for each handler
    file_handler.setLevel(level)
    console_handler.setLevel(level)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
