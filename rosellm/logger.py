import logging
from logging import Logger

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
_FORMAT = (f"%(levelname)s "
           "%(asctime)s "
           "%(filename)s:"
           "%(lineno)d] "
           "%(message)s")

"""
The date format is used to format the date and time 
in the log message.
Example: 01-01 12:00:00
"""
_DATE_FORMAT = "%m-%d %H:%M:%S"

DEFAULT_LOGGING_CONFIG = {
    "formatters": {
        "rosellm": {
            "class": "rosellm.logger.NewLineFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
    },
    "handlers": {
        "rosellm": {
            "class": "logging.StreamHandler",
            "formatter": "rosellm",
            "level": "DEBUG",
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

def NewLineFormatter():
    return logging.Formatter(_FORMAT, _DATE_FORMAT)

def _config_root_logger():
    logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)

_config_root_logger()

def init(name: str) -> Logger:
    # getLogger returns a logger with the specified name, 
    # creating it if necessary.
    # If no name is specified, return the root logger.
    return logging.getLogger(name)

"""
If the current file is run directly, __name__ = "__main__".
If the current file is imported as module, __name__ = module name.
"""
logger = init(__name__)
