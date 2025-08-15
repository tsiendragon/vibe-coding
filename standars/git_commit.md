# Commit Standards

## Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

## Types
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Formatting (no code change)
- **refactor**: Code restructuring (no behavior change)
- **perf**: Performance improvements
- **test**: Adding/updating tests
- **chore**: Maintenance tasks

## Rules
- **Subject**: Under 50 chars, imperative mood, no caps, no period
- **Body**: Explain what and why, not how (optional)
- **Footer**: Breaking changes or issue references (optional)

## Examples

**Simple:**
```bash
feat: add user authentication
fix: resolve memory leak
docs: update API guide
```

**With scope:**
```bash
feat(auth): add OAuth2 support
fix(api): handle null responses
test(utils): add date helpers tests
```

**Full format:**
```bash
fix(auth): prevent race condition in signup

Added mutex lock to ensure atomic email validation.
Previous implementation allowed duplicate emails.

Fixes #156
```

## Best Practices
- **One commit = one logical change**
- **Write in imperative**: "add" not "added"
- **Reference issues**: "Fixes #123"
- **Keep atomic**: Don't mix unrelated changes