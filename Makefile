# Makefile for RoseLLM - Python Build and Test Automation
# Usage: make [target]

# Keep documentation and literature tooling on the project's pinned environment
# even when the caller has not activated it. Override for CI or another venv with
# `make PROJECT_PYTHON=/path/to/python <target>`.
PROJECT_PYTHON ?= .venv/bin/python

.PHONY: help install dev test lint format clean build docs-lint docs docs-render docs-serve \
	research-install research-validate research-doc-audit research-equivalence-audit \
	research-test research-check \
	research-download research-recover research-recover-openalex \
	research-recover-arxiv research-recover-pmc research-promote-recovery \
	research-apply-recovery research-extract research-library research-report

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
	@echo "  make docs-lint   - Validate Markdown and TeX source"
	@echo "  make docs        - Validate and generate documentation"
	@echo "  make docs-render - Render every page and formula in headless Chrome"
	@echo "  make research-install  - Install the literature extraction dependency"
	@echo "  make research-validate - Validate and audit the literature inventories"
	@echo "  make research-doc-audit - Verify inventory/coverage representation in lab docs"
	@echo "  make research-equivalence-audit - Verify publication/preprint owner links"
	@echo "  make research-test     - Run the offline literature workflow tests"
	@echo "  make research-check    - Run all offline corpus validation, audit, and tests"
	@echo "  make research-download - Resume downloading all public corpus PDFs"
	@echo "  make research-recover  - Run all automated recovery passes in order"
	@echo "  make research-recover-openalex - Probe OpenAlex OA routes for failed downloads"
	@echo "  make research-recover-arxiv - Strict-title arXiv recovery for non-blog gaps"
	@echo "  make research-recover-pmc - Recover DOI papers through Europe PMC/PMC AWS"
	@echo "  make research-promote-recovery - Dry-run verified recovery promotion"
	@echo "  make research-apply-recovery - Apply the fully verified promotion plan"
	@echo "  make research-extract  - Extract searchable text from downloaded PDFs"
	@echo "  make research-library  - Build the readable copy-on-write PDF library"
	@echo "  make research-report   - Rebuild the coverage ledger and bibliography"

# Install package in production mode
install:
	pip install -e .

# Install package in development mode with all dependencies
dev:
	pip install -e ".[dev,test,research]"
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

## Validate Markdown and TeX source without network recovery or downloads
docs-lint: research-doc-audit
	$(PROJECT_PYTHON) scripts/check_docs_math.py

## Generate and validate documentation
docs: docs-lint
	$(PROJECT_PYTHON) -m mkdocs build --strict --clean
	$(PROJECT_PYTHON) scripts/check_docs_math.py --site-dir site

## Render every generated page and formula in a real browser
docs-render: docs
	$(PROJECT_PYTHON) scripts/check_docs_math.py --site-dir site --browser

## Serve documentation locally
docs-serve:
	$(PROJECT_PYTHON) -m mkdocs serve -a 0.0.0.0:8000

## Install the small dependency used by the standalone literature extractor
research-install:
	$(PROJECT_PYTHON) -m pip install -r requirements-research.txt

## Validate schema/duplicates and print artifact coverage
research-validate:
	$(PROJECT_PYTHON) scripts/literature_corpus.py validate
	$(PROJECT_PYTHON) scripts/literature_corpus.py audit

## Verify that every inventory ID/primary URL and index coverage row is documented
research-doc-audit:
	$(PROJECT_PYTHON) scripts/literature_doc_audit.py

## Verify every same-org recovered arXiv owner conflict has an explicit work link
research-equivalence-audit:
	$(PROJECT_PYTHON) scripts/literature_equivalence_audit.py

## Run the deterministic, network-free tests for discovery, recovery, and reporting
research-test:
	$(PROJECT_PYTHON) -m pytest tests/test_literature_*.py

## Run every offline corpus gate; safe for local checks and CI
research-check: research-validate research-doc-audit research-equivalence-audit research-test

## Resume all publicly available PDF downloads; artifacts are Git-ignored
research-download: research-validate
	$(PROJECT_PYTHON) scripts/literature_corpus.py download --workers 6

## Run the automated recovery passes sequentially to avoid archive write races
research-recover: research-validate
	$(PROJECT_PYTHON) scripts/literature_oa_recovery.py
	$(PROJECT_PYTHON) scripts/literature_secondary_recovery.py --include-missing \
		--threshold 0.96 \
		--batch-size 10 --workers 6 --delay 1 --timeout 60 --retries 2 \
		--type technical_report \
		--type research_paper \
		--type system_card \
		--type model_card \
		--type dataset \
		--type benchmark \
		--type other
	$(PROJECT_PYTHON) scripts/literature_pmc_recovery.py

## Recover failed downloads from alternate OpenAlex open-access locations
research-recover-openalex: research-validate
	$(PROJECT_PYTHON) scripts/literature_oa_recovery.py

## Search failed/missing non-blog artifacts by strict arXiv title identity
research-recover-arxiv: research-validate
	$(PROJECT_PYTHON) scripts/literature_secondary_recovery.py --include-missing \
		--threshold 0.96 \
		--batch-size 10 --workers 6 --delay 1 --timeout 60 --retries 2 \
		--type technical_report \
		--type research_paper \
		--type system_card \
		--type model_card \
		--type dataset \
		--type benchmark \
		--type other

## Recover missing DOI research papers through Europe PMC and PMC AWS
research-recover-pmc: research-validate
	$(PROJECT_PYTHON) scripts/literature_pmc_recovery.py

## Validate all recovered artifacts and preview inventory field promotion
research-promote-recovery: research-validate
	$(PROJECT_PYTHON) scripts/literature_recovery_promotion.py

## Apply the same fully validated plan, then revalidate the resulting inventories
research-apply-recovery: research-validate
	$(PROJECT_PYTHON) scripts/literature_recovery_promotion.py --apply
	$(PROJECT_PYTHON) scripts/literature_corpus.py validate
	$(PROJECT_PYTHON) scripts/literature_corpus.py audit

## Extract searchable UTF-8 text from every locally available PDF
research-extract: research-install research-validate
	$(PROJECT_PYTHON) scripts/literature_corpus.py extract --repair-quality --workers 6

## Build a Git-ignored copy-on-write view without renaming canonical artifacts
research-library: research-validate
	$(PROJECT_PYTHON) scripts/literature_library.py build

## Generate tracked human- and machine-readable coverage outputs
research-report: research-validate
	$(PROJECT_PYTHON) scripts/literature_report.py

# Run continuous integration tests (what CI/CD would run)
ci: lint test

# Check if package can be installed
check-install: build
	pip install dist/*.whl
	python -c "import rosellm; print('Import successful!')"
	pip uninstall -y rosellm
