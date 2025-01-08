"""Environment configuration module for RoseLLM.

This module provides environment variable configuration through a dataclass,
making it easy to manage and access configuration settings throughout the application.
"""

import os
from dataclasses import dataclass
from typing import Optional

"""
@dataclass is a decorator provided by Python's dataclasses module that
automatically adds generated special methods such as __init__(), __repr__(),
__eq__() to user-defined classes.

Key features:
1. Automatically generates __init__() method to initialize class attributes.
2. Provides __repr__() for nice string representation of class instances.
3. Implements __eq__() for comparing instances.
4. Reduces boilerplate code for data classes.
5. Supports type hints and default values for fields.
6. Can be customized with various parameters like frozen=True for immutable
   instances.

Example usage:
@dataclass
class Point:
    x: int = 0    # Field with type annotation and default value
    y: int = 0

p = Point(1, 2)   # Creates instance with x=1, y=2
"""


@dataclass
class Env:
    """Environment variables configuration.

    This class uses dataclass to manage environment variables with proper typing
    and default values. It provides a convenient way to access configuration
    settings throughout the application.

    Attributes:
        host: The host IP address for server.
        port: The port number for server.
        rpc_base_path: The base path for RPC service.
        api_key: The API key for authentication.
        logging_level: The logging level.
        logging_prefix: The logging prefix.
        logging_config_path: The path to logging config file.
        cache_root: The cache root directory.
        config_root: The config root directory.
        debug: Whether to enable debug mode.
        rpc_timeout: The timeout in seconds for RPC calls.
        target_device: The target device for computation.
    """

    # The host IP address for server.
    host: str = "127.0.0.1"

    # The port number for server.
    port: int = 8000

    # The base path for RPC service.
    rpc_base_path: str = "/tmp/llm"

    # The API key for authentication.
    api_key: Optional[str] = None

    # The logging level.
    logging_level: str = "INFO"

    # The logging prefix.
    logging_prefix: str = ""

    # The path to logging config file.
    logging_config_path: Optional[str] = None

    # The cache root directory.
    cache_root: str = os.path.expanduser("~/.cache/rosellm")

    # The config root directory.
    config_root: str = os.path.expanduser("~/.config/rosellm")

    # Whether to enable debug mode.
    debug: bool = False

    # The timeout in seconds for RPC calls.
    rpc_timeout: int = 60

    # The target device.
    target_device: str = "cuda"

    @classmethod
    def from_env(cls) -> "Env":
        """Create an Env instance from environment variables.

        Returns:
            An Env instance with values loaded from environment variables.
        """
        # Create instance with default values.
        env = cls()

        # Override values from environment variables if they exist.
        for field in env.__dataclass_fields__:
            env_key = f"ROSELLM_{field.upper()}"
            if env_key in os.environ:
                env_value = os.environ[env_key]
                field_type = type(getattr(env, field))
                if field_type == bool:
                    setattr(env, field, env_value.lower() in ("true", "1", "yes"))
                elif field_type == int:
                    setattr(env, field, int(env_value))
                else:
                    setattr(env, field, env_value)
        return env


# Create a global instance of Env.
env = Env.from_env()
