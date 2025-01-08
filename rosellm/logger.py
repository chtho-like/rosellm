"""Logging configuration module for RoseLLM.

This module provides logging functionality with custom formatting and configuration
options. It includes a custom formatter that properly handles multi-line log messages
and provides a consistent logging interface throughout the application.
"""

import logging
import logging.config
from logging import Logger

from rosellm.envs import env


class NewLineFormatter(logging.Formatter):
    """A custom formatter that handles newlines in log messages properly.

    This formatter ensures that each line in a multi-line log message
    is properly formatted with timestamp and other prefix information.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record as text.

        Args:
            record: A LogRecord instance containing all the information
                   pertinent to the event being logged.

        Returns:
            The formatted string.
        """
        # Get the formatted message from parent formatter.
        message = super().format(record)

        # If message contains newlines, format each line.
        if "\n" in message:
            # Get the prefix (everything before the actual message).
            prefix = message.split(record.message)[0]
            # Split message into lines.
            lines = message.splitlines()
            # First line already has prefix, add prefix to subsequent lines.
            formatted = "\n".join(
                [lines[0]] + [f"{prefix}{line}" for line in lines[1:]]
            )
            return formatted

        return message


"""
The format of the log message.
Example: DEBUG 01-01 12:00:00 rosellm.py:12] Hello, world!
%(xx)s indicates outputting xx as a string.
%(xx)d indicates outputting xx as a decimal integer.
levelname: DEBUG, INFO, WARNING, ERROR, CRITICAL
# asctime is ascii time.
asctime: 01-01 12:00:00
filename: rosellm.py
lineno: 12
message: Hello, world!
"""
_FORMAT = "%(levelname)-8s %(asctime)s %(filename)s:%(lineno)d] %(message)s"

"""
The date format is used to format the date and time
in the log message.
Example: 01-01 12:00:00
"""
_DATE_FORMAT = "%m-%d %H:%M:%S"

DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "rosellm": {
            "()": NewLineFormatter,
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
    },
    "handlers": {
        "rosellm": {
            "class": "logging.StreamHandler",
            "formatter": "rosellm",
            "level": env.logging_level,
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "rosellm": {
            "handlers": ["rosellm"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}


def _config_root_logger() -> None:
    """Configure the root logger with the default configuration."""
    logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)


_config_root_logger()


def init(name: str) -> Logger:
    """Initialize a logger with the given name.

    Args:
        name: The name of the logger to initialize.

    Returns:
        A Logger instance configured with the application's settings.
    """
    return logging.getLogger(name)


# Create a logger instance for this module.
logger = init(__name__)
