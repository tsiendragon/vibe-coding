# FastAPI Backend API Project

This is a FastAPI-based backend project using an AI-assisted development workflow with specialized agents.

## 🚀 Project Start Guide

### Step 1: Understand Your API Requirements
**Before starting any development, Claude Code needs to understand your API requirements.**

**Where are your requirements?**
- **Option A**: You have API specifications - Please provide the document, OpenAPI spec, or requirements
- **Option B**: You want to discuss API needs - We'll clarify your endpoints, data models, and business logic
- **Option C**: You have an existing API to migrate/improve - Please provide current API documentation
- **Option D**: You have a specific use case - Describe your application and we'll design the API

**Important**: The API Product Manager agent will work with you to ensure 100% requirement clarity before any development begins.

### Step 2: AI Agent Workflow Overview
This project uses a sophisticated multi-agent collaboration system where specialized AI agents work together:

1. **🎯 API Product Manager** - Clarifies API requirements and writes specifications
2. **🏗️ Tech Lead** - Designs system architecture and coordinates development
3. **📐 API Architect** - Designs RESTful APIs and data models
4. **⚡ Backend Developer** - Implements APIs with production-grade FastAPI code
5. **🗄️ Database Engineer** - Designs data models and optimizes queries
6. **🚀 DevOps Engineer** - Handles deployment, CI/CD, and infrastructure
7. **🧪 API Tester** - Creates API tests and validates functionality
8. **🔍 Code Reviewer** - Ensures code quality and security standards

## 📋 Essential Documents to Review

Before starting development, **Claude Code must thoroughly understand these key documents**:

### Workflow and Process
- **`docs/workflows/api_development_workflow.md`** - Complete API development workflow with microservice patterns
  - *Purpose*: Understand API development lifecycle, service coordination, and deployment pipeline
  - *Why Critical*: Defines the entire development process from API design to production deployment

### Agent Responsibilities
- **`.claude/agents/`** - Individual agent roles and responsibilities for API development
  - *Purpose*: Each agent specializes in different aspects of backend development
  - *Why Critical*: Ensures proper task assignment and collaboration for complex backend systems

### Development Standards
- **`docs/standards/pycode_standards.md`** - Python coding standards (NO dummy functions, use logging, fail fast)
- **`docs/standards/fastapi_standards.md`** - FastAPI-specific development practices and patterns
- **`docs/standards/pytest_stands.md`** - API testing standards (90% coverage minimum)
- **`docs/standards/git_commit_std.md`** - Git commit format and practices
  - *Purpose*: Maintain consistent, high-quality backend code
  - *Why Critical*: Ensures scalable, maintainable, and secure API development

### Document Templates
- **`docs/templates/API_SPEC/`** - API specification and endpoint documentation templates
- **`docs/templates/SYSTEM_DESIGN/`** - System architecture and microservice design templates
- **`docs/templates/DATABASE_DESIGN/`** - Database schema and migration templates
- **`docs/templates/DEPLOYMENT/`** - Infrastructure and deployment configuration templates
  - *Purpose*: Standardized formats for all API project documentation
  - *Why Critical*: Ensures consistent API design and system documentation

### Knowledge Management
- **`docs/knowledge/best_practices/`** - FastAPI patterns, microservice best practices, performance optimization
- **`docs/knowledge/error_cases/`** - Common API issues, security vulnerabilities, and solutions
  - *Purpose*: Learn from past backend development experience
  - *Why Critical*: Accelerates development and improves system reliability

## 🎯 Development Commands

### Development Server
```bash
# Start development server with auto-reload
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Production server with Gunicorn
gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Database Operations
```bash
# Database migrations
alembic upgrade head                                    # Apply migrations
alembic revision --autogenerate -m "description"       # Generate migration

# Database initialization
python -m src.db.init_db    # Initialize database with initial data
```

### Testing and Quality
```bash
# Run all tests
pytest tests/ -v

# API testing with coverage
pytest tests/api/ --cov=src --cov-report=term-missing

# Code quality checks  
black .                    # Format code
ruff check . --fix        # Lint and fix
mypy .                    # Type checking

# Security scanning
bandit -r src/            # Security vulnerability scan
```

### API Documentation
```bash
# Interactive API docs available at:
# http://localhost:8000/docs          (Swagger UI)
# http://localhost:8000/redoc         (ReDoc)
```

## 🔄 AI Development Workflow

### Phase 1: API Requirements & Specification (API Product Manager leads)
1. **API requirement confirmation** - Iterative clarification of endpoints and business logic
2. **API specification creation** - OpenAPI spec with detailed endpoint documentation
3. **Stakeholder alignment** - Tech Lead and API Architect review

### Phase 2: System Design & Architecture (Tech Lead + API Architect)
1. **System architecture design** - Microservice patterns, service boundaries
2. **API design** - RESTful endpoints, data models, authentication strategy
3. **Database design** - Schema design, relationship modeling, performance considerations

### Phase 3: Infrastructure & Database (DevOps + Database Engineer)
1. **Infrastructure setup** - Docker, databases, external services
2. **Database implementation** - Schema creation, migrations, seed data
3. **CI/CD pipeline** - Automated testing, building, and deployment

### Phase 4: API Implementation (Backend Developer leads)
1. **Core API development** - Endpoint implementation with business logic
2. **Authentication & authorization** - Security middleware and user management
3. **Data validation & serialization** - Pydantic models and request/response handling

### Phase 5: Testing & Quality Assurance (API Tester + Code Reviewer)
1. **API testing** - Unit tests, integration tests, end-to-end API tests
2. **Performance testing** - Load testing, response time optimization
3. **Security validation** - Authentication testing, input validation, vulnerability scanning

### Phase 6: Deployment & Monitoring (DevOps Engineer leads)
1. **Production deployment** - Container orchestration, environment management
2. **Monitoring setup** - Logging, metrics, health checks, alerting
3. **Documentation finalization** - API docs, deployment guides, operational runbooks

## ⚠️ Critical Guidelines

### For Claude Code
1. **NEVER start coding without clear API requirements** - Always engage API Product Manager first
2. **Follow RESTful principles** - Use proper HTTP methods, status codes, and resource naming
3. **Security first approach** - Implement authentication, authorization, and input validation from the start
4. **Use FastAPI best practices** - Dependency injection, async/await, Pydantic models
5. **Database design matters** - Plan schema carefully with performance and scalability in mind

### Development Standards
- ✅ **NO dummy endpoints** - Use `raise HTTPException(501, "Not implemented")` for placeholders
- ✅ **Proper error handling** - Return appropriate HTTP status codes with structured error responses
- ✅ **Input validation** - Use Pydantic models for all request/response validation
- ✅ **Async best practices** - Use async/await for I/O operations
- ✅ **90% test coverage minimum** - All endpoints must have comprehensive tests

## 🏗️ FastAPI Project Structure
```
src/
├── api/v1/endpoints/      # API route handlers
├── core/                  # Configuration, security, database
├── models/                # SQLAlchemy models
├── schemas/               # Pydantic schemas
├── services/              # Business logic
├── utils/                 # Utilities
├── middleware/            # Custom middleware
└── main.py               # FastAPI application entry
```

## 🤝 Getting Started

1. **Tell us about your API**: What backend services do you need?
2. **Provide requirements**: Share your API specifications, business requirements, or let's discuss
3. **Agent activation**: The API Product Manager will lead requirement clarification
4. **Follow the workflow**: Each agent will contribute their expertise in sequence
5. **Quality delivery**: Receive a production-ready FastAPI backend with full documentation

**Ready to start? Please provide your API requirements or let's discuss what backend services you want to build!**