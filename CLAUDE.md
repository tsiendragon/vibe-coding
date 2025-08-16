# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Development workspace for AI-assisted software engineering with comprehensive standards, workflows, and multi-agent collaboration patterns. Supports PyTorch ML projects, web development, backend services, and Android development.

## High-Level Architecture

### Multi-Agent Collaboration System
The repository implements an AI agent workflow (`pytorch_project/workflow/extract_rewrite/extract_rewrite_workflow.md`) where specialized agents collaborate through distinct phases:

1. **Requirements Phase**: Product Manager agent defines PRDs
2. **Design Phase**: Tech Lead coordinates architecture with Researcher input  
3. **Prototype Phase**: Algorithm Engineer builds core functionality
4. **Development Phase**: Modular implementation with continuous code review
5. **Testing Phase**: QA Engineer ensures comprehensive quality coverage
6. **Delivery Phase**: Documentation and multi-perspective validation

### Agent Coordination Pattern
- **Decision Points**: Tech Lead serves as coordinator at major milestones
- **Quality Chain**: Code Reviewer provides continuous monitoring 
- **Knowledge Flow**: Agents update shared knowledge base in `docs/knowledge/`
- **Artifact Handoffs**: Each phase produces specific documents that trigger next phase

## Development Commands

### Project Setup
```bash
# Create new AI-assisted project with multi-agent workflow
./setup_project.sh <project_name> <project_type> [target_dir]

# Supported project types:
# - pytorch: PyTorch deep learning projects
# - backend: FastAPI backend API projects
# - android: Flutter mobile application projects
# - web: Web development projects  
# - data-science: Data science projects
# - research: Research projects
# - general: General software projects

# Examples:
./setup_project.sh my-classifier pytorch
./setup_project.sh my-api backend
./setup_project.sh my-app android
./setup_project.sh web-app web ~/projects/
./setup_project.sh data-analysis data-science
```

### Testing
```bash
# Run all tests with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test layers  
pytest tests/unit/        # Unit tests only
pytest tests/integration/ # Integration tests
pytest tests/e2e/        # End-to-end tests

# Test with specific coverage threshold
pytest --cov=src --cov-fail-under=90
```

### Code Quality
```bash
# Format Python code
black .

# Lint Python code
ruff check .
ruff check . --fix    # Auto-fix issues

# Type checking
mypy .

# Combined quality check
black . && ruff check . && mypy .
```

### Backend/FastAPI Development
```bash
# Start development server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Database migrations
alembic upgrade head                                    # Apply migrations
alembic revision --autogenerate -m "description"       # Generate migration

# API documentation
# Visit http://localhost:8000/docs for interactive docs

# Production server
gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Android/Flutter Development  
```bash
# Run development build
flutter run

# Run on specific device
flutter devices                    # List available devices
flutter run -d <device_id>        # Run on specific device

# Build and test
flutter build apk                  # Build Android APK
flutter build ios                  # Build iOS (macOS only)
flutter test                      # Run all tests
flutter test test/unit/           # Run unit tests only

# Code generation
flutter packages pub run build_runner build --delete-conflicting-outputs
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

# For projects created with setup script
python -m src.lightning_module_template
```

## Critical Standards

### Code Quality (`pytorch_project/standards/pycode_standards.md`)
- NO dummy functions - raise NotImplementedError
- NO print statements - use logging
- NO emojis in code/logs
- FAIL FAST - minimize try/except blocks
- Commit after each module completion

### Testing (`pytorch_project/standards/pytest_stands.md`)  
- 90% coverage minimum
- Mirror source structure in tests/
- No `time.sleep()` in unit tests
- Use `tmp_path` for file operations
- Test naming: `test_<target>_<behavior>[_<condition>]`

### Git Commits (`pytorch_project/standards/git_commit_std.md`)
- Format: `<type>(<scope>): <subject>`
- Types: feat, fix, docs, test, refactor, perf, chore
- One commit = one logical change

### Backend/FastAPI Structure (`backend_project/standards/fastapi_standards.md`)
```
src/
├─ api/v1/endpoints/    # API route handlers
├─ core/               # Configuration, security, database
├─ models/             # SQLAlchemy models
├─ schemas/            # Pydantic schemas
├─ services/           # Business logic
├─ utils/              # Utilities
├─ middleware/         # Custom middleware
└─ main.py            # FastAPI application entry
```

### Android/Flutter Structure (`android_project/standards/flutter_standards.md`)
```
lib/
├─ main.dart          # Application entry
├─ app/               # App configuration and routes
├─ core/              # Constants, errors, network, storage
├─ data/              # Data sources, models, repositories
├─ domain/            # Entities, repositories, use cases
├─ presentation/      # Pages, widgets, providers
└─ shared/            # Extensions, mixins, validators
```

### PyTorch Structure (`pytorch_project/standards/pytorch_standards.md`)
```
src/
├─ data/          # Datasets, datamodules
├─ models/        # Networks, losses, metrics  
├─ callbacks/     # Lightning callbacks
├─ utils/         # Utilities
├─ model.py       # LightningModule
├─ train.py       # Training entry
└─ predict.py     # Inference
```

## Agent Responsibilities

### Core Agents

#### PyTorch Project Agents
- **Product Manager**: PRD creation, requirements validation
- **Tech Lead**: Architecture decisions, coordination, delivery approval
- **Researcher**: Literature review, feasibility analysis, theory validation
- **Algorithm Engineer**: Model implementation, experiments, core development
- **Code Reviewer**: Continuous quality monitoring, standards enforcement
- **QA Engineer**: Test creation, performance benchmarks, robustness testing
- **Docs Writer**: Documentation system, knowledge integration

#### Backend Project Agents
- **API Product Manager**: API requirements, endpoint specifications
- **Tech Lead**: System architecture, coordination
- **API Architect**: API design, REST standards
- **Backend Developer**: API implementation, business logic
- **Database Engineer**: Data modeling, query optimization
- **DevOps Engineer**: Deployment, CI/CD, monitoring
- **API Tester**: API testing, performance testing
- **Code Reviewer**: Code quality, security review

#### Android Project Agents
- **Mobile Product Manager**: Mobile app requirements, user experience
- **Tech Lead**: Mobile architecture, coordination
- **UI/UX Designer**: Interface design, user experience
- **Flutter Developer**: App implementation, state management
- **State Manager**: State architecture, data flow
- **Mobile Tester**: UI testing, device compatibility
- **Release Manager**: App store deployment, versioning
- **Code Reviewer**: Code quality, Flutter best practices

### Collaboration Points
- TECH_SPEC review: 4-agent validation
- Prototype evaluation: 3-agent assessment
- Module development: Engineer-Reviewer-QA triad
- Final delivery: All-agent acceptance

## Knowledge Management

### Best Practices (`docs/knowledge/best_practices/`)
- Code patterns from successful implementations
- Collaboration patterns between agents
- Technical solutions to common problems

### Error Cases (`docs/knowledge/error_cases/`)
- Common issues and resolutions
- Debugging strategies
- Performance bottlenecks

### Documentation Templates (`pytorch_project/documentation/`)
- PRD, TECH_SPEC, PROTOTYPE templates
- Test strategy and acceptance criteria
- Research and feasibility templates

## Configuration

### Claude Code Settings (`.claude/settings.local.json`)
```json
{
  "permissions": {
    "defaultMode": "acceptEdits",
    "allow": ["WebFetch(domain:github.com)"]
  }
}
```

### Agent Configuration (`.claude/agents/`)
The repository includes project-specific agent configurations:

#### PyTorch Project Agents (`pytorch_project/agents/`)
- `agent-product-manager.md` - Requirements analysis and PRD writing
- `agent-tech-lead.md` - Technical architecture and project coordination
- `agent-researcher.md` - Literature review and feasibility analysis
- `agent-algorithm-engineer.md` - Core implementation and prototyping
- `agent-code-reviewer.md` - Code quality and standards enforcement
- `agent-qa-engineer.md` - Testing and quality assurance
- `agent-docs-writer.md` - Documentation and knowledge integration

#### Backend Project Agents (`backend_project/agents/`)
- `agent-api-product-manager.md` - API requirements and specifications
- `agent-tech-lead.md` - System architecture coordination
- `agent-api-architect.md` - API design and REST standards
- `agent-backend-developer.md` - API implementation and business logic
- `agent-database-engineer.md` - Data modeling and optimization
- `agent-devops-engineer.md` - Deployment and infrastructure
- `agent-api-tester.md` - API testing and performance validation
- `agent-code-reviewer.md` - Code quality and security review

#### Android Project Agents (`android_project/agents/`)
- `agent-mobile-product-manager.md` - Mobile app requirements
- `agent-tech-lead.md` - Mobile architecture coordination
- `agent-ui-designer.md` - Interface and user experience design
- `agent-flutter-developer.md` - Flutter app implementation
- `agent-state-manager.md` - State management architecture
- `agent-mobile-tester.md` - Mobile testing and compatibility
- `agent-release-manager.md` - App store deployment
- `agent-code-reviewer.md` - Flutter code quality and best practices

### OmegaConf Pattern
- Use `class_path + init_args` for instantiation
- Support dotlist overrides
- Store in `configs/` directory

### Project Structure Templates
Projects created via `setup_project.sh` follow consistent patterns:
```
project_name/
├── src/               # Source code
├── tests/             # Test suite (unit/integration/e2e)
├── docs/              # Documentation and templates
├── configs/           # Configuration files
├── .claude/agents/    # Agent configurations
└── requirements.txt   # Dependencies
```

## Working with This Repository

### Quick Start for New Projects
1. Use the setup script to create a new project with the multi-agent workflow:
   ```bash
   ./setup_project.sh my-project pytorch
   cd my-project
   ```

2. Follow the AI-assisted development workflow as outlined in `pytorch_project/workflow/extract_rewrite/workflow.md`

3. Use the provided templates and standards from this repository

### Key Workflow Phases
1. **Requirements Phase**: Use agent-product-manager for PRD creation
2. **Design Phase**: agent-tech-lead coordinates with agent-researcher for TECH_SPEC
3. **Prototype Phase**: agent-algorithm-engineer builds core functionality
4. **Development Phase**: Modular implementation with continuous code review
5. **Testing Phase**: agent-qa-engineer ensures comprehensive quality coverage
6. **Delivery Phase**: Multi-agent validation and documentation

### Standards Enforcement

#### PyTorch Projects
- All Python code must follow `pytorch_project/standards/pycode_standards.md`
- All tests must follow `pytorch_project/standards/pytest_stands.md`
- All commits must follow `pytorch_project/standards/git_commit_std.md`
- PyTorch projects must follow `pytorch_project/standards/pytorch_standards.md`

#### Backend Projects
- All Python code must follow `backend_project/standards/pycode_standards.md`
- All FastAPI code must follow `backend_project/standards/fastapi_standards.md`
- All tests must follow `backend_project/standards/pytest_stands.md`
- All commits must follow `backend_project/standards/git_commit_std.md`

#### Android Projects
- All Dart code must follow `android_project/standards/dart_standards.md`
- All Flutter code must follow `android_project/standards/flutter_standards.md`
- All tests must follow `android_project/standards/dart_test_standards.md`
- All commits must follow `android_project/standards/git_commit_std.md`

### Knowledge Management

#### Project-Specific Knowledge Bases
- **PyTorch**: `pytorch_project/knowledge/best_practices/` and `pytorch_project/knowledge/error_cases/`
- **Backend**: `backend_project/knowledge/best_practices/` and `backend_project/knowledge/error_cases/`
- **Android**: `android_project/knowledge/best_practices/` and `android_project/knowledge/error_cases/`

#### Available Templates
- **PyTorch**: `pytorch_project/documentation/` - ML-specific templates
- **Backend**: `backend_project/documentation/` - API-specific templates
- **Android**: `android_project/documentation/` - Mobile-specific templates