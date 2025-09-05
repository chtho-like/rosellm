#!/usr/bin/env python3
"""
Test runner for all RoseLLM tests.
"""

import os
import sys
import unittest

# Add parent directory to path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_all_tests():
    """Load and run all tests in the tests directory."""
    # Find all test modules
    test_loader = unittest.TestLoader()

    # Discover tests in all subdirectories
    test_suite = test_loader.discover(start_dir=".", pattern="test_*.py")

    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)

    # Return non-zero exit code if tests failed
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
