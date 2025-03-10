"""Logging configuration module for RoseLLM.

This module provides logging functionality with custom formatting and configuration
options. It includes a custom formatter that properly handles multi-line log messages
and provides a consistent logging interface throughout the application.
"""

import logging
import logging.config
import sys
from logging import Logger

from rosellm.envs import env


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output.

    These codes can be used to color text in terminal output on systems
    that support ANSI escape sequences.
    """

    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"


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


class ColoredFormatter(logging.Formatter):
    """A formatter that adds colors to log messages based on their level.

    This formatter uses ANSI escape sequences to add color to log messages
    in the terminal, making different log levels visually distinct.
    """

    # Define colors for different log levels
    LEVEL_COLORS = {
        "DEBUG": Colors.CYAN,
        "INFO": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED,
        "CRITICAL": Colors.RED + Colors.BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the record with appropriate colors.

        Args:
            record: A LogRecord instance containing all the information
                   pertinent to the event being logged.

        Returns:
            The formatted string with color codes.
        """
        # Save the original levelname
        levelname = record.levelname

        # Add color codes
        if levelname in self.LEVEL_COLORS:
            record.levelname = (
                f"{self.LEVEL_COLORS[levelname]}{levelname}{Colors.RESET}"
            )

        # Use the parent class format method
        return super().format(record)


class ColoredNewLineFormatter(ColoredFormatter):
    """A custom formatter that handles newlines in log messages properly and adds colors.

    This formatter ensures that each line in a multi-line log message
    is properly formatted with timestamp and other prefix information,
    and also adds color coding based on log level.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record as text with colors.

        Args:
            record: A LogRecord instance containing all the information
                   pertinent to the event being logged.

        Returns:
            The formatted string with colors and proper newline handling.
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
asctime: 2025-03-10 12:00:00
filename: rosellm.py
lineno: 12
message: Hello, world!
"""
_FORMAT = "%(levelname)-8s %(asctime)s %(filename)s:%(lineno)d %(message)s"

"""
The date format is used to format the date and time
in the log message.
Example: 2025-03-10 12:00:00
"""
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "rosellm": {
            "()": ColoredNewLineFormatter,  # Use the new colored formatter
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

# Check if running in a non-interactive terminal, if so disable colors
if not sys.stdout.isatty():
    # If not in an interactive terminal, replace with non-colored formatter
    DEFAULT_LOGGING_CONFIG["formatters"]["rosellm"]["()"] = NewLineFormatter
    _config_root_logger()
