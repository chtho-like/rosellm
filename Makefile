# Makefile for RoseLLM - Python Build and Test Automation
# Usage: make [target]

.PHONY: help install dev test lint format clean build docs

# Default target
help:
	@echo "RoseLLM Build and Test Commands:"
	@echo "  make install    - Install package in production mode"
	@echo "  make dev        - Install package in development mode with all dev dependencies"
	@echo "  make test       - Run all tests"
	@echo "  make test-fast  - Run tests without slow/GPU tests"
	@echo "  make test-gpu   - Run GPU tests only"
	@echo "  make test-dist  - Run distributed tests"
	@echo "  make lint       - Run code quality checks (flake8, mypy)"
	@echo "  make format     - Auto-format code (black, isort)"
	@echo "  make clean      - Remove build artifacts and cache"
	@echo "  make build      - Build distribution packages"
	@echo "  make docs       - Generate documentation"

# Install package in production mode
install:
	pip install -e .

# Install package in development mode with all dependencies
dev:
	pip install -e ".[dev,test]"
	pre-commit install

# Run all tests
test:
	CUDA_VISIBLE_DEVICES="" pytest tests/ -v --cov=rosellm --cov-report=term-missing -m "not gpu"

test-no-cuda:
	CUDA_VISIBLE_DEVICES="" pytest tests/ -v --cov=rosellm --cov-report=term-missing -m "not gpu"

# Run tests without slow/GPU tests
test-fast:
	pytest tests/ -v -m "not slow and not gpu and not distributed"

# Run GPU tests only (requires GPU)
test-gpu:
	pytest tests/ -v -m "gpu" --gpu-count=2

# Run distributed tests
test-dist:
	pytest tests/ -v -m "distributed"

# Run specific test file
test-file:
	@read -p "Enter test file path: " filepath; \
	pytest $$filepath -v

# Run linting and type checking
lint:
	flake8 rosellm/ tests/
	mypy rosellm/ --ignore-missing-imports
	black --check rosellm/ tests/
	isort --check-only rosellm/ tests/

tlp: format test lint pre-commit

pre-commit:
	pre-commit run --all-files

# Auto-format code
format:
	black rosellm/ tests/ examples/
	isort rosellm/ tests/ examples/

# Clean build artifacts and cache
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Build distribution packages
build: clean
	python -m build

# Generate documentation
docs:
	cd docs && make html

# Run continuous integration tests (what CI/CD would run)
ci: lint test

# Check if package can be installed
check-install: build
	pip install dist/*.whl
	python -c "import rosellm; print('Import successful!')"
	pip uninstall -y rosellm