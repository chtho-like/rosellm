#!/usr/bin/env python
"""
Code Validation Script - Ensures files won't show red in IDE
This script checks for common issues that cause IDE errors before and after editing.
"""

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


class CodeValidator:
    """Validates Python code to prevent IDE errors"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.errors = []
        self.warnings = []

    def validate_file(self, filepath: Path) -> bool:
        """Validate a single Python file"""
        print(f"\n🔍 Validating: {filepath}")

        # 1. Check syntax
        if not self._check_syntax(filepath):
            return False

        # 2. Check imports
        if not self._check_imports(filepath):
            return False

        # 3. Check type hints
        self._check_type_hints(filepath)

        # 4. Run static analysis
        self._run_static_analysis(filepath)

        # 5. Check for common issues
        self._check_common_issues(filepath)

        return len(self.errors) == 0

    def _check_syntax(self, filepath: Path) -> bool:
        """Check Python syntax"""
        try:
            with open(filepath, "r") as f:
                ast.parse(f.read())
            print("  ✅ Syntax: Valid")
            return True
        except SyntaxError as e:
            self.errors.append(f"Syntax error in {filepath}:{e.lineno}: {e.msg}")
            print(f"  ❌ Syntax: {e}")
            return False

    def _check_imports(self, filepath: Path) -> bool:
        """Check if all imports can be resolved"""
        with open(filepath, "r") as f:
            tree = ast.parse(f.read())

        import_errors = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not self._can_import(alias.name):
                        import_errors.append(f"Cannot import '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if not self._can_import(module, alias.name):
                        import_errors.append(
                            f"Cannot import '{alias.name}' from '{module}'"
                        )

        if import_errors:
            self.errors.extend(import_errors)
            print(f"  ❌ Imports: {len(import_errors)} errors")
            for err in import_errors[:3]:  # Show first 3
                print(f"     - {err}")
            return False
        else:
            print("  ✅ Imports: All resolved")
            return True

    def _can_import(self, module: str, name: Optional[str] = None) -> bool:
        """Check if a module/name can be imported"""
        try:
            # Try to find the module
            spec = importlib.util.find_spec(module)
            if spec is None:
                return False

            # If checking specific name, try to import and check
            if name:
                try:
                    mod = importlib.import_module(module)
                    return hasattr(mod, name)
                except (ImportError, AttributeError):
                    return False
            return True
        except (ImportError, ModuleNotFoundError):
            return False

    def _check_type_hints(self, filepath: Path) -> None:
        """Check for type hint issues using mypy"""
        try:
            result = subprocess.run(
                [
                    "mypy",
                    str(filepath),
                    "--ignore-missing-imports",
                    "--no-error-summary",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                # Parse mypy output
                errors = [
                    line for line in result.stdout.split("\n") if "error:" in line
                ]
                if errors:
                    self.warnings.extend(errors[:3])  # First 3 errors
                    print(f"  ⚠️  Type hints: {len(errors)} issues")
                else:
                    print("  ✅ Type hints: Clean")
            else:
                print("  ✅ Type hints: Clean")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("  ⏭️  Type hints: Skipped (mypy not available)")

    def _run_static_analysis(self, filepath: Path) -> None:
        """Run flake8 static analysis"""
        try:
            result = subprocess.run(
                [
                    "flake8",
                    str(filepath),
                    "--max-line-length=100",
                    "--ignore=E203,W503,E501",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                issues = result.stdout.strip().split("\n")
                self.warnings.extend(issues[:3])
                print(f"  ⚠️  Linting: {len(issues)} issues")
            else:
                print("  ✅ Linting: Clean")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("  ⏭️  Linting: Skipped (flake8 not available)")

    def _check_common_issues(self, filepath: Path) -> None:
        """Check for common issues that cause IDE errors"""
        with open(filepath, "r") as f:
            content = f.read()
            lines = content.split("\n")

        issues = []

        # Check for undefined variables
        tree = ast.parse(content)
        undefined_vars = self._find_undefined_vars(tree)
        if undefined_vars:
            issues.append(
                f"Potentially undefined variables: {', '.join(undefined_vars[:3])}"
            )

        # Check for missing return statements
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.returns and not self._has_return(node):
                    issues.append(f"Function '{node.name}' missing return statement")

        # Check for None checks
        for i, line in enumerate(lines, 1):
            if "None." in line or "None[" in line:
                issues.append(f"Line {i}: Potential None access without check")

        if issues:
            self.warnings.extend(issues)
            print(f"  ⚠️  Common issues: {len(issues)} found")
        else:
            print("  ✅ Common issues: None found")

    def _find_undefined_vars(self, tree: ast.AST) -> List[str]:
        """Find potentially undefined variables"""
        # Simplified check - in real implementation would be more sophisticated
        defined = set()
        used = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    defined.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    used.add(node.id)

        # Exclude builtins and common imports
        builtins = {
            "True",
            "False",
            "None",
            "int",
            "str",
            "float",
            "list",
            "dict",
            "set",
            "tuple",
            "print",
            "len",
            "range",
            "open",
            "super",
            "self",
        }
        undefined = used - defined - builtins

        return list(undefined)[:5]  # Return first 5

    def _has_return(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has return statement"""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                return True
        return False

    def validate_directory(self, directory: Path, pattern: str = "*.py") -> bool:
        """Validate all Python files in directory"""
        files = list(directory.rglob(pattern))
        print(f"\n{'='*60}")
        print(f"Validating {len(files)} Python files in {directory}")
        print("=" * 60)

        all_valid = True
        for filepath in files:
            if "__pycache__" in str(filepath):
                continue
            if not self.validate_file(filepath):
                all_valid = False

        return all_valid

    def print_summary(self):
        """Print validation summary"""
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print("=" * 60)

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10
                print(f"  - {error}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:10]:  # Show first 10
                print(f"  - {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All files are clean! No IDE errors expected.")
        elif not self.errors:
            print("\n✅ No critical errors. Files should not appear red in IDE.")
            print("⚠️  Some warnings exist but won't cause red errors.")
        else:
            print("\n❌ Critical errors found. Files will appear red in IDE.")
            print("Fix the errors above before committing.")


def validate_before_edit(filepath: str) -> Dict:
    """Validate file and store state before editing"""
    validator = CodeValidator()
    is_valid = validator.validate_file(Path(filepath))

    state = {
        "filepath": filepath,
        "was_valid": is_valid,
        "errors_before": validator.errors.copy(),
        "warnings_before": validator.warnings.copy(),
    }

    # Save state to temp file
    state_file = Path(f"/tmp/validation_state_{Path(filepath).name}.json")
    with open(state_file, "w") as f:
        json.dump(state, f)

    return state


def validate_after_edit(filepath: str) -> bool:
    """Validate file after editing and compare with before state"""
    # Load previous state
    state_file = Path(f"/tmp/validation_state_{Path(filepath).name}.json")
    if state_file.exists():
        with open(state_file, "r") as f:
            before_state = json.load(f)
    else:
        before_state = {"was_valid": True, "errors_before": [], "warnings_before": []}

    # Validate current state
    validator = CodeValidator()
    is_valid = validator.validate_file(Path(filepath))

    # Compare
    print(f"\n{'='*60}")
    print("BEFORE/AFTER COMPARISON")
    print("=" * 60)

    if before_state["was_valid"] and is_valid:
        print("✅ File was valid before and remains valid after edit")
    elif not before_state["was_valid"] and is_valid:
        print("✅ File was invalid before but is now valid! Great job!")
    elif before_state["was_valid"] and not is_valid:
        print("❌ File was valid before but now has errors!")
        print("New errors introduced:")
        for error in validator.errors:
            if error not in before_state["errors_before"]:
                print(f"  - {error}")
    else:
        print("⚠️  File had issues before and still has issues")

    # Clean up state file
    if state_file.exists():
        state_file.unlink()

    return is_valid


def main():
    """Main validation workflow"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python validate_code.py <file_or_directory>")
        print("  python validate_code.py --before-edit <file>")
        print("  python validate_code.py --after-edit <file>")
        print("  python validate_code.py --all")
        sys.exit(1)

    if sys.argv[1] == "--before-edit" and len(sys.argv) > 2:
        _ = validate_before_edit(sys.argv[2])  # state
        print("\n📸 State saved. Edit the file and run with --after-edit")

    elif sys.argv[1] == "--after-edit" and len(sys.argv) > 2:
        is_valid = validate_after_edit(sys.argv[2])
        sys.exit(0 if is_valid else 1)

    elif sys.argv[1] == "--all":
        validator = CodeValidator()
        is_valid = validator.validate_directory(Path("rosellm"))
        validator.print_summary()
        sys.exit(0 if is_valid else 1)

    else:
        path = Path(sys.argv[1])
        validator = CodeValidator()

        if path.is_file():
            is_valid = validator.validate_file(path)
        else:
            is_valid = validator.validate_directory(path)

        validator.print_summary()
        sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
