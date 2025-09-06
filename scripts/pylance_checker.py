#!/usr/bin/env python3
"""
Pylance/Type Checker Helper Script.

This script helps identify and fix common Pylance/type checking issues:
1. Missing required parameters
2. Type mismatches
3. Import issues
"""

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class PylanceChecker:
    """Check and fix common Pylance issues."""

    def __init__(self, project_root: Path = Path.cwd()):
        """Initialize the checker."""
        self.project_root = project_root
        self.issues: List[Dict] = []

    def run_mypy(self, file_path: Path) -> List[str]:
        """Run mypy on a file and return errors."""
        try:
            result = subprocess.run(
                ["mypy", "--ignore-missing-imports", str(file_path)],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            return result.stdout.split("\n") if result.returncode != 0 else []
        except Exception as e:
            print(f"Error running mypy: {e}")
            return []

    def run_pyright(self, file_path: Path) -> Dict:
        """Run pyright/pylance on a file and return diagnostics."""
        try:
            result = subprocess.run(
                ["pyright", "--outputjson", str(file_path)],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            if result.returncode == 0:
                return {}
            output = json.loads(result.stdout)
            return output.get("generalDiagnostics", [])
        except Exception as e:
            print("Note: pyright not installed. Install with: npm install -g pyright")
            return {}

    def check_function_calls(self, file_path: Path) -> List[Dict]:
        """Check for missing parameters in function calls."""
        issues = []
        try:
            with open(file_path, "r") as f:
                tree = ast.parse(f.read(), filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Check if it's a class instantiation
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        # Common config classes that might have issues
                        if "Config" in func_name:
                            issues.append(
                                {
                                    "type": "missing_params",
                                    "function": func_name,
                                    "line": node.lineno,
                                    "file": str(file_path),
                                    "hint": f"Check if {func_name} requires additional parameters",
                                }
                            )
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        return issues

    def suggest_fixes(self, issues: List[Dict]) -> List[str]:
        """Suggest fixes for identified issues."""
        fixes = []
        for issue in issues:
            if issue["type"] == "missing_params":
                fixes.append(
                    f"Line {issue['line']}: Add missing parameters to {issue['function']}. "
                    "Check the class definition for required fields without defaults."
                )
        return fixes

    def check_file(self, file_path: Path) -> Tuple[List[str], List[str]]:
        """Check a file for Pylance/type issues."""
        print(f"\nChecking {file_path}...")

        # Run mypy
        mypy_errors = self.run_mypy(file_path)

        # Check function calls
        call_issues = self.check_function_calls(file_path)

        # Generate fixes
        fixes = self.suggest_fixes(call_issues)

        return mypy_errors, fixes

    def auto_fix_config_calls(self, file_path: Path) -> bool:
        """Automatically add common default parameters to Config calls."""
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Common fixes for TrainingConfig
            if "TrainingConfig(" in content:
                # Check if these parameters are missing
                if "max_steps=" not in content:
                    content = content.replace(
                        "TrainingConfig(",
                        "TrainingConfig(\n        max_steps=None,  # Auto-added default\n        ",
                    )
                if "warmup_steps=" not in content:
                    content = content.replace(
                        "TrainingConfig(",
                        "TrainingConfig(\n        warmup_steps=0,  # Auto-added default\n        ",
                    )
                if "log_interval=" not in content:
                    content = content.replace(
                        "TrainingConfig(",
                        "TrainingConfig(\n        log_interval=10,  # Auto-added default\n        ",
                    )
                if "eval_interval=" not in content:
                    content = content.replace(
                        "TrainingConfig(",
                        "TrainingConfig(\n        eval_interval=100,  # Auto-added default\n        ",
                    )

            # Write back
            with open(file_path, "w") as f:
                f.write(content)

            print(f"✅ Auto-fixed config calls in {file_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to auto-fix {file_path}: {e}")
            return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Check and fix Pylance/type issues")
    parser.add_argument(
        "files", nargs="*", help="Files to check (default: all Python files)"
    )
    parser.add_argument("--fix", action="store_true", help="Attempt to auto-fix issues")
    parser.add_argument(
        "--check-all", action="store_true", help="Check all Python files in project"
    )
    args = parser.parse_args()

    checker = PylanceChecker()

    # Determine files to check
    if args.check_all:
        files = list(Path(".").rglob("*.py"))
        # Exclude venv, .conda, etc.
        files = [
            f
            for f in files
            if not any(p in str(f) for p in [".conda", "venv", "__pycache__", "build"])
        ]
    elif args.files:
        files = [Path(f) for f in args.files]
    else:
        # Default: check examples and main source
        files = list(Path("examples").glob("*.py")) + list(
            Path("rosellm").rglob("*.py")
        )

    print(f"Checking {len(files)} files...")

    total_errors = []
    total_fixes = []

    for file_path in files:
        errors, fixes = checker.check_file(file_path)

        if errors:
            total_errors.extend(errors)
            print(f"  ❌ {len(errors)} mypy errors")
            for error in errors[:3]:  # Show first 3
                print(f"    {error}")

        if fixes:
            total_fixes.extend(fixes)
            print(f"  ⚠️  {len(fixes)} potential issues")
            for fix in fixes:
                print(f"    {fix}")

        if args.fix and fixes:
            if checker.auto_fix_config_calls(file_path):
                print("  ✅ Applied auto-fixes")

        if not errors and not fixes:
            print("  ✅ No issues found")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total mypy errors: {len(total_errors)}")
    print(f"  Total potential issues: {len(total_fixes)}")

    if total_errors or total_fixes:
        print("\nTo fix issues:")
        print("  1. Run with --fix flag to apply auto-fixes")
        print("  2. Review remaining issues manually")
        print("  3. Add type hints where needed")
        print("  4. Use Optional[] for nullable parameters")
        return 1
    else:
        print("\n✅ All files pass type checking!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
