# Claude CLI Python Examples

This directory contains Python examples for calling Claude Code CLI programmatically.

## Files

- `claude_cli_wrapper.py` - Main wrapper class and simple function for calling Claude CLI
- `claude_cli_examples.py` - Advanced examples and use cases

## Installation

Make sure Claude Code CLI is installed and available in your PATH:
```bash
which claude  # Should show the path to claude executable
```

## Basic Usage

### Simple Function Call

```python
from claude_cli_wrapper import simple_claude_prompt

# Basic prompt
result = simple_claude_prompt("What is 2 + 2?")
print(result)  # Output: 4

# With web search permission
result = simple_claude_prompt(
    "What's the weather today?", 
    allow_web_search=True,
    output_json=True
)
print(result)

# With JSON output
result = simple_claude_prompt(
    "List 3 Python tips", 
    output_json=True
)
print(result)
```

### Using the ClaudeCLI Class

```python
from claude_cli_wrapper import ClaudeCLI

# Initialize
cli = ClaudeCLI()

# Simple prompt with JSON output
result = cli.prompt(
    "Explain recursion",
    output_format="json",
    max_turns=1
)

# With permissions and multiple turns
result = cli.prompt(
    "Research the latest AI news",
    output_format="json",
    max_turns=3,
    allow=["WebSearch"],
    permission_mode="bypassPermissions"
)

# Pipe input to Claude
code = "def hello(): print('world')"
result = cli.pipe_input(
    code,
    "Review this Python code",
    output_format="text"
)
```

## Advanced Examples

Run the examples file to see various use cases:

```bash
python3 examples/claude_cli_examples.py
```

### Available Functions

1. **Code Review**: Analyze code files for bugs, performance, and style
2. **Git Commit Messages**: Generate commit messages from git diff
3. **Error Explanation**: Get detailed explanations and fixes for errors
4. **Batch Processing**: Process multiple prompts efficiently
5. **File References**: Use @ syntax to reference files
6. **Refactoring**: Get code refactoring suggestions

## Permission Management

### Method 1: Command Line Arguments
```python
result = cli.prompt(
    "Search for news",
    allow=["WebSearch", "WebFetch"]  # Uses --allowedTools internally
)
```

### Method 2: Permission Mode
```python
result = cli.prompt(
    "Do something",
    permission_mode="bypassPermissions"  # Skip all permission checks
)
```

### Method 3: Configure in settings.json
Create `~/.claude/settings.json`:
```json
{
  "permissions": {
    "allow": ["WebSearch", "WebFetch"]
  }
}
```

## Output Formats

### Text Output (Default)
```python
result = cli.prompt("Hello", output_format="text")
print(result["result"])  # Plain text response
```

### JSON Output
```python
result = cli.prompt("Hello", output_format="json")
# Returns structured data with metadata:
# - result: The actual response
# - total_cost_usd: Cost of the API call
# - usage: Token usage details
# - session_id: Unique session identifier
```

## Error Handling

```python
result = cli.prompt("Something")

if result.get("type") == "error":
    print(f"Error: {result['error']}")
elif result.get("returncode") != 0:
    print(f"Command failed: {result['stderr']}")
else:
    print(f"Success: {result['result']}")
```

## Best Practices

1. **Use JSON output** for programmatic processing
2. **Set max_turns** to control Claude's iterations
3. **Handle permissions** appropriately for your use case
4. **Add delays** between calls to avoid rate limiting
5. **Limit input size** when piping large texts
6. **Check return codes** and handle errors gracefully

## Common Use Cases

### CI/CD Integration
```python
# In your CI pipeline
result = cli.prompt(
    "Run tests and report results",
    output_format="json",
    max_turns=3,
    permission_mode="bypassPermissions"
)

if result["returncode"] == 0:
    data = json.loads(result["stdout"])
    if "error" in data["result"].lower():
        sys.exit(1)
```

### Automated Code Review
```python
# Pre-commit hook
changes = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True)
result = cli.pipe_input(
    changes.stdout,
    "Review these changes for issues",
    output_format="json"
)
```

### Documentation Generation
```python
# Generate docs from code
result = cli.prompt(
    f"@{source_file} Generate API documentation",
    output_format="text",
    max_turns=1
)
with open("API.md", "w") as f:
    f.write(result["result"])
```