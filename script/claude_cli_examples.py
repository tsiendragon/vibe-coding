#!/usr/bin/env python3
"""
Advanced examples of using Claude CLI from Python.
Demonstrates various use cases and patterns.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import time


def claude_code_review(file_path: str) -> Dict[str, Any]:
    """
    Use Claude to review a code file.
    
    Args:
        file_path: Path to the code file to review
        
    Returns:
        Review results from Claude
    """
    # Read the file content
    with open(file_path, 'r') as f:
        code = f.read()
    
    prompt = f"""Review this code and provide:
1. Potential bugs or issues
2. Performance improvements
3. Code style suggestions
4. Security concerns

Code:
{code}
"""
    
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--max-turns", "1"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        return json.loads(result.stdout)
    else:
        return {"error": result.stderr}


def claude_git_commit_message() -> str:
    """
    Generate a git commit message based on current changes.
    
    Returns:
        Suggested commit message
    """
    # Get git diff
    diff_result = subprocess.run(
        ["git", "diff", "--cached"],
        capture_output=True,
        text=True
    )
    
    if not diff_result.stdout:
        # Try unstaged changes
        diff_result = subprocess.run(
            ["git", "diff"],
            capture_output=True,
            text=True
        )
    
    if not diff_result.stdout:
        return "No changes detected"
    
    prompt = f"""Based on these git changes, write a concise commit message following conventional commit format (type(scope): description):

{diff_result.stdout[:3000]}  # Limit diff size
"""
    
    cmd = ["claude", "-p", prompt, "--max-turns", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return result.stdout.strip() if result.returncode == 0 else "Error generating commit message"


def claude_explain_error(error_message: str, context: str = "") -> str:
    """
    Ask Claude to explain an error and suggest fixes.
    
    Args:
        error_message: The error message to explain
        context: Additional context about when the error occurred
        
    Returns:
        Claude's explanation and suggestions
    """
    prompt = f"""Explain this error and provide solutions:

Error: {error_message}

Context: {context if context else 'No additional context provided'}

Please provide:
1. What causes this error
2. Step-by-step solution
3. How to prevent it in the future
"""
    
    cmd = ["claude", "-p", prompt, "--max-turns", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return result.stdout if result.returncode == 0 else "Error getting explanation"


def claude_batch_analysis(prompts: List[str], parallel: bool = False) -> List[Dict[str, Any]]:
    """
    Process multiple prompts with Claude.
    
    Args:
        prompts: List of prompts to process
        parallel: Whether to process in parallel (be careful with rate limits)
        
    Returns:
        List of results
    """
    results = []
    
    if parallel:
        import concurrent.futures
        
        def process_prompt(prompt):
            cmd = ["claude", "-p", prompt, "--output-format", "json", "--max-turns", "1"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return json.loads(result.stdout)
            return {"error": result.stderr}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(process_prompt, prompts))
    else:
        for prompt in prompts:
            cmd = ["claude", "-p", prompt, "--output-format", "json", "--max-turns", "1"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                results.append(json.loads(result.stdout))
            else:
                results.append({"error": result.stderr})
            
            # Small delay to avoid rate limiting
            time.sleep(1)
    
    return results


def claude_with_file_reference(file_path: str, question: str) -> str:
    """
    Ask Claude a question about a specific file using @ reference.
    
    Args:
        file_path: Path to the file
        question: Question about the file
        
    Returns:
        Claude's response
    """
    # Note: This assumes Claude can access the file path
    prompt = f"@{file_path} {question}"
    
    cmd = ["claude", "-p", prompt, "--max-turns", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(file_path).parent)
    
    return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"


def claude_interactive_session(initial_prompt: str, max_interactions: int = 3):
    """
    Simulate an interactive session with Claude.
    
    Args:
        initial_prompt: Starting prompt
        max_interactions: Maximum number of back-and-forth interactions
    """
    print(f"Starting interactive session (max {max_interactions} turns)")
    print("=" * 50)
    
    current_prompt = initial_prompt
    
    for i in range(max_interactions):
        print(f"\nTurn {i+1}")
        print(f"User: {current_prompt}")
        print("-" * 30)
        
        cmd = ["claude", "-p", current_prompt, "--max-turns", "1"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            break
        
        response = result.stdout.strip()
        print(f"Claude: {response}")
        
        if i < max_interactions - 1:
            # Get next prompt from user
            current_prompt = input("\nYour response (or 'quit' to exit): ")
            if current_prompt.lower() == 'quit':
                break
    
    print("\nSession ended")


def claude_refactor_suggestion(code: str, language: str = "python") -> str:
    """
    Get refactoring suggestions for code.
    
    Args:
        code: The code to refactor
        language: Programming language
        
    Returns:
        Refactored code with explanations
    """
    prompt = f"""Refactor this {language} code for better readability, performance, and maintainability:

```{language}
{code}
```

Provide:
1. The refactored code
2. Explanation of changes
3. Why each change improves the code
"""
    
    cmd = ["claude", "-p", prompt, "--max-turns", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return result.stdout if result.returncode == 0 else "Error getting refactoring suggestions"


if __name__ == "__main__":
    # Example 1: Generate commit message
    print("Example 1: Git Commit Message Generation")
    print("-" * 50)
    # commit_msg = claude_git_commit_message()
    # print(f"Suggested commit: {commit_msg}")
    print("(Skipped - requires git changes)")
    print()
    
    # Example 2: Explain an error
    print("Example 2: Error Explanation")
    print("-" * 50)
    error = "TypeError: 'NoneType' object is not subscriptable"
    explanation = claude_explain_error(
        error,
        "Occurred when trying to access data['key'] after API call"
    )
    print(f"Explanation: {explanation[:500]}...")
    print()
    
    # Example 3: Batch processing
    print("Example 3: Batch Processing")
    print("-" * 50)
    questions = [
        "What is the time complexity of quicksort?",
        "Explain the difference between TCP and UDP in one sentence",
        "What is a Python decorator?"
    ]
    
    print("Processing batch questions...")
    results = claude_batch_analysis(questions, parallel=False)
    for i, (q, r) in enumerate(zip(questions, results), 1):
        if "result" in r:
            print(f"{i}. Q: {q}")
            print(f"   A: {r['result'][:100]}...")
        else:
            print(f"{i}. Error processing: {q}")
    print()
    
    # Example 4: Code refactoring
    print("Example 4: Code Refactoring")
    print("-" * 50)
    sample_code = """
def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            result.append(data[i] * 2)
    return result
"""
    
    refactored = claude_refactor_suggestion(sample_code)
    print(f"Refactoring suggestion: {refactored[:500]}...")
    
    # Example 5: Interactive session (commented out to avoid blocking)
    # print("\nExample 5: Interactive Session")
    # print("-" * 50)
    # claude_interactive_session("Let's discuss Python best practices", max_interactions=2)