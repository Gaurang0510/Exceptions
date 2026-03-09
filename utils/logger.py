"""
Logging Utilities
==================
Centralised logging configuration for the entire project.
"""

import logging
import sys
from pathlib import Path

from config.settings import LOG_LEVEL, LOG_FORMAT, LOG_DIR


def setup_logging(name: str = "news2trade") -> logging.Logger:
    """Configure and return a logger with console + file handlers."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
