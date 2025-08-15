# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a development workspace focused on PyTorch/Lightning ML projects with comprehensive standards and workflow automation. The repository contains standardized templates, coding standards, and AI agent workflows for building machine learning systems.

## Project Structure

The repository is organized into several key areas:

- `pytorch_project/`: Main PyTorch project workspace with standards and templates
- `claude_code/`: Claude Code configuration and AI agent workflows
- `pytorch_project/standards/`: Comprehensive coding and testing standards
- `pytorch_project/documentation/`: PRD and technical specification templates
- `pytorch_project/workflow/`: Automated workflows for common ML development tasks

## Development Standards

### PyTorch Project Structure
Follow the standardized directory layout as defined in `pytorch_project/standards/pytorch_standards.md`:

```
src/
├─ data/          # Data processing, augmentation, datasets, datamodules
├─ models/        # Network architectures, loss, metrics, optimizers, schedulers
├─ callbacks/     # PyTorch Lightning callbacks
├─ utils/         # Math, text, image, tensor utilities
├─ tools/         # Additional tooling and utilities
├─ model.py       # Main LightningModule
├─ train.py       # Training entry point with OmegaConf
└─ predict.py     # Prediction script
```

### Configuration Management
- Use OmegaConf with `class_path + init_args` pattern for all instantiable objects
- Support dotlist overrides for flexible parameter tuning
- Store configurations in `configs/` directory
- Use absolute paths or relative to project root

### Code Quality Standards
Based on `pytorch_project/standards/pycode_stands.md`:
- MUST NOT create dummy functions
- MUST raise errors immediately in critical code paths
- MUST use logging instead of print statements
- MUST NOT use emojis in logging/print
- MUST commit after completing each module
- Minimize try/except usage - fail fast approach

### Testing Standards
Comprehensive pytest standards from `pytorch_project/standards/pytest_stands.md`:

#### Test Structure
- Mirror source structure: `src/x/y.py` → `tests/unit/x/test_y.py`
- Three test layers:
  - `unit/`: Pure logic, no I/O or network
  - `integration/`: Local dependencies (DB, files)
  - `e2e/`: Real external services

#### Coverage & Naming
- Minimum 90% line and branch coverage required
- Functions: `test_<target>_<behavior>[_<condition>]`
- MUST create test file for every new module
- MUST include happy path + error case per public function

#### Critical Rules
- MUST NOT use `time.sleep()` in unit tests
- MUST NOT make network calls without `@pytest.mark.integration`
- MUST use `tmp_path` fixture for file I/O tests

### Git Commit Standards
Follow conventional commits as specified in `pytorch_project/standards/git_commit_std.md`:
- Format: `<type>(<scope>): <subject>`
- Types: feat, fix, docs, style, refactor, perf, test, chore
- Subject: Under 50 chars, imperative mood, no caps, no period
- One commit = one logical change

## Development Commands

### Testing
```bash
# Run all tests with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test layers
pytest tests/unit/      # Unit tests only
pytest tests/integration/  # Integration tests
pytest tests/e2e/      # End-to-end tests

# Verbose output
pytest -v
```

### PyTorch Training
```bash
# Basic training
python -m src.train --config configs/default.yaml

# With overrides
python -m src.train --config configs/default.yaml \
  trainer.max_epochs=100 \
  trainer.devices=2 \
  data.params.batch_size=128

# Fast development run
python -m src.train --config configs/default.yaml --fast_dev_run=1
```

### Code Quality (when configured)
```bash
black .              # Format code
ruff check .         # Lint code  
mypy .              # Type checking
```

## AI Agent Integration

This repository includes AI agent workflows in `claude_code/`:
- Algorithm engineering workflows
- Automated code generation and testing
- PRD and technical specification management
- Multi-agent collaboration patterns

## Documentation Templates

Use standardized templates from `pytorch_project/documentation/`:
- PRD templates with review checklists
- Technical specification templates
- Clear separation of product and technical concerns