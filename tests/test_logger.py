"""Test module for the logger functionality.

This module contains tests for the logger functionality, including:
1. Logger initialization
2. Custom formatter behavior
3. Log level handling
4. Multi-line message formatting
"""

import io
import logging

import pytest

from rosellm.logger import NewLineFormatter, init

"""
This module contains tests for the logger functionality.

The tests verify:
1. The logger initialization works correctly.
2. The NewLineFormatter formats messages properly.
3. Different log levels work as expected.

The module uses pytest fixtures to set up test environments. The main fixture
'test_logger' provides a logger instance with captured output for testing
log messages. This allows us to verify both the formatting and content of
the log output.
"""


@pytest.fixture
def test_logger():
    """Fixture that provides a test logger with captured output.

    Returns:
        tuple: (logger instance, StringIO buffer for captured output).
    """
    # Capture stdout to test log output.
    stdout = io.StringIO()
    handler = logging.StreamHandler(stdout)
    test_logger = init("test")
    formatter = NewLineFormatter("%(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.DEBUG)

    yield test_logger, stdout, handler

    # Cleanup
    stdout.close()
    test_logger.removeHandler(handler)


def test_init_logger():
    """Test that init() returns a proper logger instance."""
    test_logger = init("test_logger")
    assert isinstance(test_logger, logging.Logger)
    assert test_logger.name == "test_logger"


def test_newline_formatter(test_logger):
    """Test that NewLineFormatter properly formats multi-line messages."""
    logger, stdout, handler = test_logger
    formatter = NewLineFormatter("PREFIX %(message)s")
    handler.setFormatter(formatter)

    # Test single line message.
    logger.info("Single line message.")
    output = stdout.getvalue()
    assert "Single line message" in output

    # Clear buffer.
    stdout.truncate(0)
    stdout.seek(0)

    # Test multi-line message.
    logger.info("First line\nSecond line\nThird line")
    output = stdout.getvalue()
    lines = output.strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        assert line.startswith("PREFIX")


def test_log_levels(test_logger):
    """Test that different log levels work as expected."""
    logger, stdout, handler = test_logger
    test_messages = {
        "debug": "Debug message.",
        "info": "Info message.",
        "warning": "Warning message.",
        "error": "Error message.",
        "critical": "Critical message.",
    }

    for level, msg in test_messages.items():
        # Clear buffer.
        stdout.truncate(0)
        stdout.seek(0)

        # Log message at appropriate level.
        getattr(logger, level)(msg)
        output = stdout.getvalue()

        # Verify message was logged.
        assert msg in output
        # Verify log level appears in output.
        assert level.upper() in output
