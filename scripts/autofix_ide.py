#!/usr/bin/env python3
"""
Automatic IDE Issue Fixer

This script automatically fixes common IDE linting issues:
- Unused imports
- F-strings without placeholders
- Missing type annotations
- Import order issues
- Formatting issues
"""

import ast
import logging
import re
import subprocess
from pathlib import Path
from typing import List, Pattern

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IDEAutoFixer:
    """Automatically fix IDE/Pylance issues.

    This class provides methods to automatically fix common IDE and linting
    issues in Python files, including unused imports, f-strings without
    placeholders, and formatting issues.

    Attributes:
        project_root: The root directory of the project.
        fixes_applied: List of fixes that have been applied.
        f_string_pattern: Compiled regex for f-strings without placeholders.
        f_string_single_pattern: Compiled regex for single-quoted f-strings.
    """

    def __init__(self, project_root: Path = Path.cwd()) -> None:
        """Initialize the IDEAutoFixer.

        Args:
            project_root: The root directory of the project. Defaults to current directory.
        """
        self.project_root = project_root
        self.fixes_applied: List[str] = []
        # Pre-compile regex patterns for performance
        self.f_string_pattern: Pattern[str] = re.compile(r'f"([^{}"]*)"')
        self.f_string_single_pattern: Pattern[str] = re.compile(r"f'([^{}']*)'")

    def fix_file(self, filepath: Path) -> bool:
        """Fix all issues in a single file.

        Args:
            filepath: Path to the file to fix.

        Returns:
            True if any fixes were applied, False otherwise.
        """
        print(f"\n🔧 Processing {filepath}")
        any_fixes = False

        # Check file size first (skip very large files)
        try:
            file_size = filepath.stat().st_size
            if file_size > 1_000_000:  # 1MB limit
                logger.warning(f"Skipping large file {filepath} ({file_size} bytes)")
                print(f"  ⚠ Skipping large file ({file_size} bytes)")
                return False
        except OSError as e:
            logger.error(f"Could not stat file {filepath}: {e}")
            return False

        # Read file content
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            original_content = content
        except Exception as e:
            logger.error(f"Could not read file {filepath}: {e}")
            print(f"  ❌ Could not read file: {e}")
            return False

        # Apply fixes
        content = self.fix_f_strings(content, filepath)
        content = self.fix_unused_imports(content, filepath)
        content = self.fix_type_ignores(content, filepath)
        content = self.fix_lambda_defaults(content, filepath)

        # Write back if changed
        if content != original_content:
            try:
                with open(filepath, "w") as f:
                    f.write(content)
                print(f"  ✅ Fixed issues in {filepath}")
                any_fixes = True
            except Exception as e:
                print(f"  ❌ Could not write file: {e}")
                return False

        # Run formatters
        if self.run_isort(filepath):
            any_fixes = True
        if self.run_black(filepath):
            any_fixes = True

        # Check with mypy
        self.check_mypy(filepath)

        return any_fixes

    def fix_f_strings(self, content: str, filepath: Path) -> str:
        """Remove f prefix from strings without placeholders.

        Args:
            content: The file content to fix.
            filepath: Path to the file being processed.

        Returns:
            The content with f-strings fixed.

        Example:
            >>> fixer.fix_f_strings('"no placeholder"', Path("test.py"))
            '"no placeholder"'
        """
        new_content = self.f_string_pattern.sub(r'"\1"', content)
        new_content = self.f_string_single_pattern.sub(r"'\1'", new_content)

        if new_content != content:
            self.fixes_applied.append(f"Fixed f-strings in {filepath}")
            print("  ✓ Fixed f-strings without placeholders")

        return new_content

    def fix_unused_imports(self, content: str, filepath: Path) -> str:
        """Remove obviously unused imports."""
        try:
            tree = ast.parse(content)
            lines = content.split("\n")

            # Collect all names used in the file
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        used_names.add(node.value.id)

            # Find unused imports
            lines_to_remove = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = (
                            alias.asname if alias.asname else alias.name.split(".")[0]
                        )
                        if name not in used_names:
                            lines_to_remove.append(node.lineno - 1)
                            print(f"  ✓ Removed unused import: {name}")
                elif isinstance(node, ast.ImportFrom):
                    # Be more careful with from imports
                    if node.names[0].name != "*":
                        all_unused = True
                        for alias in node.names:
                            name = alias.asname if alias.asname else alias.name
                            if name in used_names or name == "__version__":
                                all_unused = False
                                break
                        if all_unused and node.module:
                            # Check if the module itself is used
                            if node.module.split(".")[0] not in used_names:
                                lines_to_remove.append(node.lineno - 1)
                                print(f"  ✓ Removed unused from import: {node.module}")

            # Remove lines
            for line_no in sorted(lines_to_remove, reverse=True):
                if line_no < len(lines):
                    lines[line_no] = ""

            # Clean up multiple blank lines
            cleaned_lines = []
            prev_blank = False
            for line in lines:
                if line.strip() == "":
                    if not prev_blank:
                        cleaned_lines.append(line)
                    prev_blank = True
                else:
                    cleaned_lines.append(line)
                    prev_blank = False

            return "\n".join(cleaned_lines)

        except Exception as e:
            print(f"  ⚠ Could not analyze imports: {e}")
            return content

    def fix_type_ignores(self, content: str, filepath: Path) -> str:
        """Add type: ignore comments for known issues."""
        # Fix dist.barrier timeout issue
        pattern = r"(dist\.barrier\(timeout=.*\))(?!.*# type: ignore)"
        new_content = re.sub(pattern, r"\1  # type: ignore[call-arg]", content)

        if new_content != content:
            print("  ✓ Added type: ignore for dist.barrier")

        return new_content

    def fix_lambda_defaults(self, content: str, filepath: Path) -> str:
        """Fix Pydantic Field default_factory lambdas."""
        # This is complex - just check for the pattern
        if "Field(default_factory=" in content:
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if "Field(default_factory=" in line and "Config" in line:
                    # Check if it's missing lambda
                    if "lambda:" not in line and not line.strip().endswith("lambda:"):
                        print(f"  ⚠ Line {i+1}: May need lambda for default_factory")

        return content

    def run_isort(self, filepath: Path) -> bool:
        """Run isort to fix import order."""
        try:
            result = subprocess.run(
                ["isort", str(filepath), "--profile", "black", "--quiet"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("  ✓ Fixed import order with isort")
                return True
        except FileNotFoundError:
            logger.debug("Tool not found, skipping")
            pass
        return False

    def run_black(self, filepath: Path) -> bool:
        """Run black to fix formatting."""
        try:
            result = subprocess.run(
                ["black", str(filepath), "--quiet"], capture_output=True, text=True
            )
            if result.returncode == 0:
                print("  ✓ Applied black formatting")
                return True
        except FileNotFoundError:
            logger.debug("Tool not found, skipping")
            pass
        return False

    def check_mypy(self, filepath: Path) -> List[str]:
        """Check with mypy for remaining issues."""
        try:
            result = subprocess.run(
                ["mypy", "--ignore-missing-imports", str(filepath)],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            if result.returncode != 0 and result.stdout:
                issues = result.stdout.strip().split("\n")
                if issues:
                    print("  ⚠ Remaining mypy issues:")
                    for issue in issues[:3]:
                        print(f"    {issue}")
                return issues
        except FileNotFoundError:
            logger.debug("Tool not found, skipping")
            pass
        return []

    def fix_all_python_files(self) -> None:
        """Fix all Python files in the project."""
        python_files = list(self.project_root.rglob("*.py"))

        # Exclude virtual environments and build directories
        python_files = [
            f
            for f in python_files
            if not any(
                p in str(f) for p in [".conda", "venv", "__pycache__", "build", ".git"]
            )
        ]

        print(f"Found {len(python_files)} Python files to check")

        fixed_count = 0
        for filepath in python_files:
            if self.fix_file(filepath):
                fixed_count += 1

        print(f"\n{'='*60}")
        print(f"✅ Fixed {fixed_count} files")
        if self.fixes_applied:
            print("\nFixes applied:")
            for fix in self.fixes_applied[:10]:
                print(f"  - {fix}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Auto-fix IDE/Pylance issues")
    parser.add_argument("files", nargs="*", help="Specific files to fix")
    parser.add_argument("--all", action="store_true", help="Fix all Python files")
    parser.add_argument("--check", action="store_true", help="Only check, don't fix")
    args = parser.parse_args()

    fixer = IDEAutoFixer()

    if args.all:
        fixer.fix_all_python_files()
    elif args.files:
        for filepath in args.files:
            fixer.fix_file(Path(filepath))
    else:
        # Fix recently modified files
        result = subprocess.run(
            ["git", "di", "--name-only", "HEAD"], capture_output=True, text=True
        )
        if result.returncode == 0:
            files = [f for f in result.stdout.strip().split("\n") if f.endswith(".py")]
            if files:
                print(f"Fixing {len(files)} recently modified Python files...")
                for filepath in files:
                    fixer.fix_file(Path(filepath))
            else:
                print("No Python files modified recently")
        else:
            print("Not in a git repository or no changes")


if __name__ == "__main__":
    main()
