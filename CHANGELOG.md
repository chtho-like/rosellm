# Changelog

All notable changes to RoseLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2024

### Added
- Constants module (`constants.py`) for centralized configuration values
- Comprehensive docstrings for factory functions in config module
- Model validator for training duration (max_steps vs num_epochs)
- Type hints throughout the codebase
- Auto-fixing tool (`autofix_ide.py`) for common IDE issues
- Pre-commit hooks for code quality
- Test suite for configuration validation

### Changed
- Made `num_epochs` optional to allow exclusive use of `max_steps`
- Default gradient clip value changed from None to 1.0 for consistency
- Replaced bare `except:` clauses with specific exception types
- Replaced magic numbers with named constants throughout codebase
- Improved import organization (stdlib → third-party → local)

### Fixed
- Fixed test failures in config validation
- Fixed typo in mixed precision test ("in" → "inf")
- Fixed contradictory assertions in autofix integration test
- Fixed max_steps/num_epochs validation conflict
- Fixed recursive validation for sub-configurations
- Fixed unused variable warnings by using `_` convention

### Security
- No hardcoded credentials or secrets
- Proper use of temporary directories for test files
- Specific exception handling instead of broad catches

## [0.1.0] - 2024

### Initial Release
- 5D Parallelism support (TP, PP, DP, CP, EP)
- Mixed precision training (FP16/BF16/FP8)
- Memory optimization techniques
- Activation checkpointing
- CPU offloading
- Distributed training support