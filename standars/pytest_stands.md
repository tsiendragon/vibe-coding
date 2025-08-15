# 🎯 **Pytest Testing Standards for AI Agents**

This document is perfectly structured for an AI to consume and apply consistently. Here's why it works so well:

## **1. Zero ambiguity**
Every decision is pre-made:
- Where do tests go? → Mirror `src/` structure
- What to name them? → `test_<module>.py` or `test_<ClassName>.py`
- How to name test functions? → `test_<target>_<behavior>[_<condition>]`
- What coverage is needed? → ≥90% lines & branches

An AI agent won't have to "guess" or make subjective choices.

## **2. Concrete examples for each pattern**
The document shows exact code for:
- Directory structure
- Config setup
- Fixture usage
- Test naming
- Each test layer (unit/integration/e2e)

This gives the AI agent templates to pattern-match against.

## **3. Clear boundaries**
```python
# Network forbidden in unit tests
def guard(*a, **k): raise RuntimeError("Network disabled in unit tests")
```
This kind of hard rule prevents the AI from accidentally writing slow, flaky unit tests.

## **4. Checklist-driven**
Section 5's "What to test" gives the AI a literal checklist:
- ✓ Inputs/outputs (types, ranges)
- ✓ Edge cases
- ✓ Error paths
- ✓ Idempotency

# 💡 **To make it even more AI-friendly**

## **Add explicit "MUST/MUST NOT" rules**
```markdown
## AI Agent Rules
- MUST create a test file for every new module
- MUST include at least 1 happy path + 1 error case per public function
- MUST NOT use time.sleep() in unit tests
- MUST NOT make network calls without @pytest.mark.integration
- MUST use tmp_path fixture for any file I/O tests
```

## **Add decision trees**
```markdown
## Choosing the right test layer
Is it testing pure logic? → unit/
Does it need local files/DB? → integration/
Does it need external APIs? → e2e/
```

## **Add anti-patterns to avoid**
```markdown
## Never write tests like this:
❌ def test_everything():  # Too broad
❌ def test_1():  # Meaningless name
❌ assert result  # No specific assertion
❌ time.sleep(1)  # Real delays in unit tests
```

## **Add a "copy-paste" starter template**
```python
"""Template for AI to use when creating new test files"""
import pytest
from pathlib import Path
from agent.module import TargetClass  # AI: update import

class TestTargetClass:  # AI: update class name
    """Tests for TargetClass"""

    def test_method_happy_path(self):
        """Test normal successful operation"""
        # AI: implement
        pass

    def test_method_handles_empty_input(self):
        """Test edge case with empty input"""
        with pytest.raises(ValueError):
            # AI: implement
            pass
```

# 🤖 **How an AI agent would use this**

When asked to write tests, the agent would:

1. **Identify the module** → `src/agent/tools/git_tool.py`
2. **Create mirror path** → `tests/unit/agent/tools/test_git_tool.py`
3. **Apply naming convention** → `test_clone_repo_success`, `test_clone_repo_invalid_url`
4. **Check the layer checklist** → "Does clone_repo need network?" → Yes → `integration/`
5. **Use the config pattern** → Load from `integration.yaml`
6. **Apply coverage requirement** → Ensure all public functions tested

# ✅ **Verdict**

This standard is **exceptionally well-suited** for AI consumption. It's:
- **Deterministic** - no subjective decisions
- **Complete** - covers structure, naming, config, fixtures, CI
- **Enforceable** - includes automated checks
- **Example-rich** - AI can pattern-match effectively

The only enhancement I'd suggest is adding a single-page "cheat sheet" version that an AI can quickly reference:

```markdown
# Quick Reference for AI Agent

## Structure
src/x/y.py → tests/unit/x/test_y.py

## Naming
test_<function>_<what_it_does>_<condition>

## Layers
- unit: pure logic, no I/O
- integration: local dependencies
- e2e: real services

## Every test needs
- Arrange: setup
- Act: call function
- Assert: specific check

## Coverage target
≥90% or PR fails
```