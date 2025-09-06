# Best Practices: Preventing Red Files in IDE

This guide helps ensure your Python files don't show red errors in IDEs like VSCode with Pylance.

## Quick Checklist Before Committing

Run these commands to validate your changes:

```bash
# 1. Validate specific file
python scripts/validate_code.py path/to/file.py

# 2. Validate all files
python scripts/validate_code.py --all

# 3. Before/after validation
python scripts/validate_code.py --before-edit file.py
# ... make your edits ...
python scripts/validate_code.py --after-edit file.py

# 4. Run pre-commit hooks
pre-commit run --all-files
```

## Common Causes of Red Files and Solutions

### 1. Import Errors

**Problem:** Incorrect import paths or missing modules
```python
# ❌ Will show red
from rosellm.non_existent import Something
from .relative_import import Class  # in wrong context
```

**Solution:** Verify imports before using
```python
# ✅ Correct
from rosellm.rosetrainer import RoseTrainer
from rosellm.rosetrainer.parallelism import parallel_state
```

**Validation:**
```bash
python -c "from rosellm.module import Class; print('OK')"
```

### 2. Type Checking Errors

**Problem:** Optional types not handled
```python
# ❌ Will show red
def process(value: Optional[int]) -> int:
    return value * 2  # Error: value might be None
```

**Solution:** Add None checks
```python
# ✅ Correct
def process(value: Optional[int]) -> int:
    if value is None:
        return 0
    return value * 2
```

### 3. Undefined Variables

**Problem:** Using variables before definition
```python
# ❌ Will show red
if condition:
    x = 10
print(x)  # Error: x might not be defined
```

**Solution:** Initialize variables
```python
# ✅ Correct
x = 0  # Default value
if condition:
    x = 10
print(x)
```

### 4. Missing Return Statements

**Problem:** Function with return type but no return
```python
# ❌ Will show red
def get_value() -> int:
    if condition:
        return 42
    # Missing return for else case
```

**Solution:** Ensure all paths return
```python
# ✅ Correct
def get_value() -> int:
    if condition:
        return 42
    return 0  # Default return
```

### 5. Class/Module Not Matching File Expectations

**Problem:** `__init__.py` importing non-existent classes
```python
# ❌ In __init__.py
from .module import NonExistentClass  # Will show red
```

**Solution:** Verify class names match
```python
# ✅ First check what exists
grep "^class" module.py

# Then import correctly
from .module import ActualClassName
```

## Automated Validation Workflow

### Step 1: Setup Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### Step 2: Configure IDE

**VSCode settings.json:**
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.linting.flake8Args": [
        "--max-line-length=100",
        "--ignore=E203,W503"
    ]
}
```

### Step 3: Use Validation Script

**Before editing:**
```bash
python scripts/validate_code.py --before-edit file.py
```

**After editing:**
```bash
python scripts/validate_code.py --after-edit file.py
```

## Testing Import Changes

When modifying imports or module structure:

```python
# test_imports.py
import sys
import importlib

def test_import(module_path):
    """Test if import works"""
    try:
        module = importlib.import_module(module_path)
        print(f"✅ {module_path}")
        return True
    except ImportError as e:
        print(f"❌ {module_path}: {e}")
        return False

# Test all critical imports
test_import("rosellm.rosetrainer")
test_import("rosellm.rosetrainer.parallelism")
test_import("rosellm.rosetrainer.parallelism.parallel_state")
```

## Type Hints Best Practices

### Use Type Guards

```python
from typing import Optional, cast

def process(value: Optional[str]) -> str:
    if value is None:
        return ""
    
    # Type guard ensures value is str here
    return value.upper()
```

### Handle Optional Process Groups

```python
def get_group() -> Optional[dist.ProcessGroup]:
    if not initialized:
        return None
    return group

# Usage with check
group = get_group()
if group is not None:
    # Now safe to use group
    dist.all_reduce(tensor, group=group)
```

## CI/CD Integration

Add to `.github/workflows/validate.yml`:

```yaml
name: Validate No IDE Errors
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install deps
        run: |
          pip install -e ".[dev]"
          
      - name: Validate no red files
        run: |
          python scripts/validate_code.py --all
          
      - name: Type check
        run: |
          mypy rosellm/ --ignore-missing-imports
```

## Quick Fixes for Common Issues

### Fix All Import Errors
```bash
# Find all import statements
grep -r "^from\|^import" rosellm/ | grep -v __pycache__

# Test each import
python -c "from module import name"
```

### Fix Type Errors
```bash
# Run mypy to find all type errors
mypy rosellm/ --ignore-missing-imports

# Add type: ignore for unfixable external issues
value = external_func()  # type: ignore
```

### Fix Undefined Variables
```bash
# Use flake8 to find undefined names
flake8 rosellm/ --select=F821
```

## Emergency Fixes

If file is red and you need quick fix:

```bash
# 1. Check syntax
python -m py_compile file.py

# 2. Check imports
python -c "import file"

# 3. Add type ignore if needed
# type: ignore

# 4. Initialize potentially undefined vars
var: Optional[Type] = None

# 5. Add return statements
return None  # or appropriate default
```

## Summary

**Before every commit:**
1. Run `python scripts/validate_code.py --all`
2. Fix any errors shown
3. Run `pre-commit run --all-files`
4. Verify no files show red in IDE

**Key principles:**
- Always handle `None` for Optional types
- Ensure all imports resolve
- Initialize variables before use
- Provide returns for all code paths
- Test imports after modifying `__init__.py`

Following these practices ensures clean, IDE-friendly code that won't show red errors.