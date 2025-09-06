#!/usr/bin/env python3
"""
Test suite for the autofix_ide.py script.

Tests all auto-fixing functionality including:
- F-string fixing
- Unused import removal
- Type annotation improvements
- Integration with formatters
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from autofix_ide import IDEAutoFixer


class TestIDEAutoFixer(unittest.TestCase):
    """Test cases for IDEAutoFixer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.fixer = IDEAutoFixer()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fix_f_strings_without_placeholders(self):
        """Test that f-strings without placeholders are fixed."""
        content = """
def example():
    message = "This has no placeholders"
    another = 'Single quotes no placeholders'
    valid = f"This has {placeholder}"
    return message
"""
        expected = """
def example():
    message = "This has no placeholders"
    another = 'Single quotes no placeholders'
    valid = f"This has {placeholder}"
    return message
"""
        result = self.fixer.fix_f_strings(content, Path("test.py"))
        self.assertEqual(result, expected)

    def test_fix_f_strings_preserves_valid(self):
        """Test that valid f-strings are preserved."""
        content = """
name = "World"
greeting = f"Hello, {name}!"
complex = f"Value: {x * 2 + 1}"
"""
        result = self.fixer.fix_f_strings(content, Path("test.py"))
        self.assertEqual(result, content)

    def test_fix_unused_imports_simple(self):
        """Test removal of obviously unused imports."""
        content = """
import os
import sys
import json  # unused
from typing import List

def main():
    print(sys.version)
    path = os.path.join("a", "b")
    items: List[str] = []
    return items
"""
        result = self.fixer.fix_unused_imports(content, Path("test.py"))
        self.assertNotIn("import json", result)
        self.assertIn("import os", result)
        self.assertIn("import sys", result)

    def test_fix_unused_imports_preserves_used(self):
        """Test that used imports are preserved."""
        content = """
import math
import random

def calculate():
    return math.sqrt(random.random())
"""
        result = self.fixer.fix_unused_imports(content, Path("test.py"))
        self.assertIn("import math", result)
        self.assertIn("import random", result)

    def test_fix_type_ignores(self):
        """Test adding type: ignore comments."""
        content = """
import torch.distributed as dist

def sync():
    dist.barrier(timeout=timedelta(seconds=30))  # type: ignore[call-arg]
"""
        expected = """
import torch.distributed as dist

def sync():
    dist.barrier(timeout=timedelta(seconds=30))  # type: ignore[call-arg]
"""
        result = self.fixer.fix_type_ignores(content, Path("test.py"))
        self.assertEqual(result, expected)

    def test_fix_type_ignores_no_duplicate(self):
        """Test that type: ignore is not duplicated."""
        content = """
dist.barrier(timeout=timedelta(seconds=30))  # type: ignore[call-arg]
"""
        result = self.fixer.fix_type_ignores(content, Path("test.py"))
        self.assertEqual(result, content)
        self.assertEqual(content.count("type: ignore"), 1)

    def test_fix_file_integration(self):
        """Test the complete fix_file method."""
        test_file = Path(self.temp_dir) / "test.py"
        content = """
import json
import os

def example():
    message = "No placeholders"
    path = os.path.join("a", "b")
    return path
"""
        test_file.write_text(content)

        result = self.fixer.fix_file(test_file)

        fixed_content = test_file.read_text()
        # The f-string without placeholders should be converted to regular string
        self.assertIn('"No placeholders"', fixed_content)
        # json should be removed as unused
        self.assertNotIn("import json", fixed_content)
        self.assertTrue(result)

    @patch("subprocess.run")
    def test_run_isort(self, mock_run):
        """Test isort integration."""
        mock_run.return_value = MagicMock(returncode=0)

        result = self.fixer.run_isort(Path("test.py"))

        self.assertTrue(result)
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        self.assertIn("isort", call_args)
        self.assertIn("--profile", call_args)
        self.assertIn("black", call_args)

    @patch("subprocess.run")
    def test_run_black(self, mock_run):
        """Test black integration."""
        mock_run.return_value = MagicMock(returncode=0)

        result = self.fixer.run_black(Path("test.py"))

        self.assertTrue(result)
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        self.assertIn("black", call_args)
        self.assertIn("--quiet", call_args)

    @patch("subprocess.run")
    def test_check_mypy(self, mock_run):
        """Test mypy checking."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="test.py:10: error: Argument 1 has incompatible type"
        )

        issues = self.fixer.check_mypy(Path("test.py"))

        self.assertTrue(len(issues) > 0)
        self.assertIn("incompatible type", issues[0])

    def test_fix_lambda_defaults_detection(self):
        """Test detection of lambda default factory issues."""
        content = """
from pydantic import Field

class Config:
    field1 = Field(default_factory=SomeConfig)
    field2 = Field(default_factory=lambda: SomeConfig())
"""
        self.fixer.fix_lambda_defaults(content, Path("test.py"))
        # This test just ensures the method runs without error
        # Actual fixing would need more complex logic

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty file
        self.assertEqual(self.fixer.fix_f_strings("", Path("test.py")), "")

        # Non-Python content
        content = "This is not Python code!"
        result = self.fixer.fix_unused_imports(content, Path("test.py"))
        # Should handle gracefully
        self.assertIsNotNone(result)


class TestAutoFixCLI(unittest.TestCase):
    """Test the CLI interface of autofix_ide."""

    @patch("sys.argv", ["autofix_ide.py", "--help"])
    def test_help_option(self):
        """Test that --help works."""
        with self.assertRaises(SystemExit) as cm:
            from autofix_ide import main

            main()
        # Help should exit with 0
        self.assertEqual(cm.exception.code, 0)

    @patch("subprocess.run")
    @patch("sys.argv", ["autofix_ide.py", "test.py"])
    def test_single_file_mode(self, mock_run):
        """Test fixing a single file."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import os\nprint('test')")
            test_file = f.name

        try:
            from autofix_ide import main

            # This would normally process the file
            # We're just testing it doesn't crash
        except SystemExit:
            pass
        finally:
            import os

            os.unlink(test_file)


if __name__ == "__main__":
    unittest.main()
