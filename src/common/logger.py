"""Standardized colored logging utility for the HRI multi-agent architecture."""

import logging
import sys
from typing import Optional


class Formatter(logging.Formatter):
    """Custom colorized formatter for CLI output."""

    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    CYAN = "\x1b[36;20m"

    FORMAT = "%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    FORMATS = {
        logging.DEBUG: GREY + FORMAT + RESET,
        logging.INFO: CYAN + FORMAT + RESET,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.ERROR: RED + FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT + RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT)
        formatter = logging.Formatter(log_fmt, datefmt=self.DATE_FORMAT)
        return formatter.format(record)


def setup_logger(
    name: str = "HRI",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Sets up and returns a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(Formatter())
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "HRI") -> logging.Logger:
    """Retrieve an existing logger or configure a default one."""
    return logging.getLogger(name)
