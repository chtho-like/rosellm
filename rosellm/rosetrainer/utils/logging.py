"""Logging utilities for RoseLLM."""

import logging
import sys
from typing import Optional

# Global logger instance
_logger: Optional[logging.Logger] = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name. If None, uses the root logger.

    Returns:
        Logger instance.
    """
    global _logger

    if _logger is None:
        # Initialize root logger
        _logger = logging.getLogger("rosellm")
        _logger.setLevel(logging.INFO)

        # Create console handler if not already present
        if not _logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            _logger.addHandler(handler)

    if name:
        return logging.getLogger(name)
    return _logger


def set_log_level(level: str):
    """Set the logging level.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logger = get_logger()
    logger.setLevel(numeric_level)
    for handler in logger.handlers:
        handler.setLevel(numeric_level)
