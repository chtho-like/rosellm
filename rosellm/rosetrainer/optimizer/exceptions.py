"""
Custom exceptions for the distributed optimizer module.

This module defines specific exception types for better error handling
and debugging in distributed training scenarios.
"""


class OptimizerError(Exception):
    """Base exception for optimizer-related errors."""

    pass


class GradientBufferError(OptimizerError):
    """Exception raised for gradient buffer operations."""

    pass


class PartitioningError(OptimizerError):
    """Exception raised for parameter partitioning errors."""

    pass


class CommunicationError(OptimizerError):
    """Exception raised for distributed communication failures."""

    pass


class ConfigurationError(OptimizerError):
    """Exception raised for invalid optimizer configuration."""

    pass


class SynchronizationError(OptimizerError):
    """Exception raised for rank synchronization issues."""

    pass
