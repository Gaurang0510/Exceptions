"""
Common Helpers
===============
Miscellaneous utility functions used across modules.
"""

import time
import functools
import logging
from typing import Callable

logger = logging.getLogger(__name__)


def timer(func: Callable) -> Callable:
    """Decorator that logs the wall-clock time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info("%s completed in %.2fs", func.__name__, elapsed)
        return result

    return wrapper


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns *default* when denominator is zero."""
    return numerator / denominator if denominator != 0 else default


def truncate(text: str, max_len: int = 200) -> str:
    """Truncate text with ellipsis."""
    return text[:max_len] + "…" if len(text) > max_len else text
