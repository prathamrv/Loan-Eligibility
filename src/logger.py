"""
logger.py
---------
Centralized logging configuration. Every module in the pipeline calls
`get_logger(__name__)` so log records are attributable to their source
and consistently formatted, both on the console and in a persistent
rotating log file under /logs.
"""

import logging
from logging.handlers import RotatingFileHandler

from src import config

_CONFIGURED = False


def _configure_root_logger() -> None:
    """Attach a console handler and a rotating file handler to the root
    logger exactly once, no matter how many modules import this file."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(config.LOG_LEVEL)

    formatter = logging.Formatter(config.LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        config.LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring shared handlers on first use."""
    _configure_root_logger()
    return logging.getLogger(name)
