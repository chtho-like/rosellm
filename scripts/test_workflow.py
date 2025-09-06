#!/usr/bin/env python
"""
Complete Testing Workflow for RoseLLM
Demonstrates build, test, and validation procedures
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print("=" * 60)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {description} - SUCCESS")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ {description} - FAILED")
        if result.stderr:
            print(f"Error: {result.stderr}")
        return False
    return True


def main():
    """Main testing workflow"""

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    print("🚀 RoseLLM Testing Workflow")
    print("=" * 60)

    # 1. Install in development mode
    if not run_command(
        "pip install -e '.[dev,test]'", "Install package in development mode"
    ):
        print("Failed to install package")
        sys.exit(1)

    # 2. Check imports
    if not run_command(
        "python -c 'from rosellm.rosetrainer import RoseTrainer; print(\"Import OK\")'",
        "Test basic imports",
    ):
        print("Import test failed")
        sys.exit(1)

    # 3. Run type checking
    run_command(
        "mypy rosellm/ --ignore-missing-imports || true", "Type checking (optional)"
    )

    # 4. Run linting
    run_command(
        "flake8 rosellm/ --max-line-length=100 --ignore=E203,W503 || true",
        "Code linting (optional)",
    )

    # 5. Run unit tests
    if not run_command(
        "pytest tests/rosetrainer/test_engine.py -v", "Unit tests for engine"
    ):
        print("Unit tests failed")

    # 6. Run parallel state tests with mocking
    if not run_command(
        "pytest tests/rosetrainer/parallelism/test_parallel_state.py -v",
        "Parallel state tests",
    ):
        print("Parallel state tests failed")

    # 7. Test with 2 GPUs if available
    run_command(
        """python -c '
import torch
if torch.cuda.device_count() >= 2:
    print(f"Found {torch.cuda.device_count()} GPUs")
    import subprocess
    subprocess.run(["torchrun", "--nproc_per_node=2", 
                   "examples/training_example.py"])
else:
    print("Less than 2 GPUs, skipping GPU test")
'""",
        "GPU parallel test (if available)",
    )

    # 8. Test with CPU parallelism
    run_command(
        "torchrun --nproc_per_node=4 --master_port=29500 examples/training_example.py",
        "CPU parallel test (4 processes)",
    )

    # 9. Generate coverage report
    run_command(
        "pytest tests/ --cov=rosellm --cov-report=term-missing --cov-report=html",
        "Generate coverage report",
    )

    # 10. Build distribution
    run_command("python -m build", "Build distribution packages")

    print("\n" + "=" * 60)
    print("✅ Testing workflow complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check coverage report: open htmlcov/index.html")
    print("2. Check built packages: ls dist/")
    print("3. Run integration tests: pytest tests/integration/")
    print("4. Test on cluster: sbatch scripts/cluster_test.sh")


if __name__ == "__main__":
    main()
