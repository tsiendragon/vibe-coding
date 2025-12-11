# Repository Guidelines

## Project Structure & Module Organization
The workspace is organized around reusable playbooks. `docs/` hosts framework specs and research; consult `docs/guide/specs/*` before proposing new patterns. Domain templates live under `projects/<domain>/` with `agents/`, `standards/`, `workflows/`, `templates/`, and a `config.yaml`. Shared scaffolding belongs in `common/`, while runnable illustrations stay in `examples/`. Python tooling and automation helpers are in `script/`. Place new assets alongside peers rather than mixing domains.

## Build, Test, and Development Commands
Use `./setup_project.sh <project_path>` to scaffold or wire an existing repo; the script auto-detects project state and prompts for domain. Local automation is exercised via `python3 script/dev_assistant.py --help` and focused demos live in `python3 script/claude_cli_examples.py`. Verify the Claude CLI integration with `which claude` before invoking automation steps.

## Coding Style & Naming Conventions
Python sources follow `projects/backend/standards/pycode_standards.md`: format with Black (line length 100), lint with Ruff, and keep `mypy --strict` clean. Structure new packages as `src/<package>/{core,adapters,services,utils}`. Use lowercase kebab-case for agents (`agent-api-tester.md`) and workflows (`extract_rewrite/workflow.yaml`). Markdown guidelines favor concise headings and code fences; keep bilingual notes consistent with surrounding files.

## Testing Guidelines
pytest is the default harness. Adopt the expectations in `<domain>/standards/pytest_stands.md`, including strict markers and fixture patterns. Target 90% coverage or higher with `pytest --cov=src --cov-report=term-missing` for units, then extend with scoped runs such as `pytest tests/integration -m integration` and `pytest tests/e2e -m e2e --env=staging`. Mark asynchronous or external interactions and mock live services. Document notable strategies in `knowledge/best_practices/`.

## Commit & Pull Request Guidelines
Commits use Conventional Commit prefixes seen in history (`feat:`, `docs:`, `test:`); include a scope when touching a specific domain (`feat(backend): ...`). Pull requests should summarize intent, link to specs or issues, list executed commands, and attach relevant artefacts (workflow diffs, CLI transcripts, screenshots). Highlight affected agents or standards so reviewers can validate downstream impacts promptly.

## Agent & Workflow Updates
When revising automation, align metadata with `docs/guide/specs/agent-specification.md` and `workflow-specification.md`. Centralize reusable conventions in `common/` instead of duplicating within domain folders. Update `examples/` to illustrate new behavior and record configuration nuances inside the corresponding domain `knowledge/` folder.
