# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This appears to be a new or minimal repository with limited structure. The only existing documentation is a comprehensive pytest testing standards guide located at `claude_code/pytest_stands.md`.

## Testing Standards

This repository follows strict pytest testing standards as documented in `claude_code/pytest_stands.md`. Key principles:

### Test Structure
- Tests mirror source structure: `src/x/y.py` → `tests/unit/x/test_y.py`
- Three test layers:
  - `unit/`: Pure logic tests, no I/O or network
  - `integration/`: Tests with local dependencies (DB, files)
  - `e2e/`: Tests with real external services

### Test Naming Convention
- Files: `test_<module>.py` or `test_<ClassName>.py`
- Functions: `test_<target>_<behavior>[_<condition>]`
- Example: `test_parse_json_handles_empty_string()`

### Coverage Requirements
- Minimum 90% line and branch coverage required
- All public functions must have tests
- Include happy path and error cases

### Key Rules for Test Development
- MUST create a test file for every new module
- MUST include at least 1 happy path + 1 error case per public function
- MUST NOT use `time.sleep()` in unit tests
- MUST NOT make network calls without `@pytest.mark.integration`
- MUST use `tmp_path` fixture for any file I/O tests

## Development Commands

Since this is a minimal repository without established build tools, commands will need to be determined as the project develops. When Python development begins, typical commands would include:

```bash
# Run tests (once pytest is configured)
pytest
pytest tests/unit/  # Run only unit tests
pytest -v  # Verbose output
pytest --cov=src --cov-report=term-missing  # With coverage

# Code quality (when configured)
black .  # Format code
ruff check .  # Lint code
mypy .  # Type checking
```

## Architecture Notes

Currently minimal - architecture will emerge as the codebase develops. Follow the testing standards document for any test-related development.