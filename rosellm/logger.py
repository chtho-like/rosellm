import logging
from logging import Logger

_FORMAT = "%(levelname)s"

def _config_root_logger():
    pass

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
