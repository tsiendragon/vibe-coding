#!/usr/bin/env python3
"""
Claude Code CLI Python Wrapper
This module provides a Python interface to call Claude Code CLI commands.
"""

import subprocess
import json
import sys
from typing import Optional, Dict, Any, List
from pathlib import Path


class ClaudeCLI:
    """Wrapper class for Claude Code CLI commands."""
    
    def __init__(self, working_dir: Optional[str] = None):
        """
        Initialize Claude CLI wrapper.
        
        Args:
            working_dir: Optional working directory for Claude commands
        """
        self.working_dir = working_dir
    
    def prompt(
        self,
        message: str,
        output_format: str = "text",
        max_turns: Optional[int] = None,
        model: Optional[str] = None,
        allow: Optional[List[str]] = None,
        permission_mode: Optional[str] = None,
        add_dirs: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a Claude prompt command.
        
        Args:
            message: The prompt message to send to Claude
            output_format: Output format ('text' or 'json')
            max_turns: Maximum number of turns for Claude to take
            model: Specific model to use
            allow: List of permissions to allow (e.g., ['WebSearch', 'Read'])
            permission_mode: Permission mode (e.g., 'bypassPermissions')
            add_dirs: Additional directories Claude can access
            **kwargs: Additional arguments to pass to the command
            
        Returns:
            Dict containing the result and metadata
        """
        cmd = ["claude", "-p", message]
        
        if output_format == "json":
            cmd.extend(["--output-format", "json"])
        
        if max_turns:
            cmd.extend(["--max-turns", str(max_turns)])
        
        if model:
            cmd.extend(["--model", model])
        
        if allow:
            cmd.extend(["--allowedTools", " ".join(allow)])
        
        if permission_mode:
            cmd.extend(["--permission-mode", permission_mode])
        
        if add_dirs:
            for directory in add_dirs:
                cmd.extend(["--add-dir", directory])
        
        # Add any additional arguments
        for key, value in kwargs.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.working_dir,
                check=False
            )
            
            if output_format == "json":
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    return {
                        "type": "error",
                        "error": "Failed to parse JSON output",
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode
                    }
            else:
                return {
                    "type": "result",
                    "result": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
                
        except Exception as e:
            return {
                "type": "error",
                "error": str(e)
            }
    
    def pipe_input(
        self,
        input_text: str,
        prompt: str,
        output_format: str = "text",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Pipe input text to Claude with a prompt.
        
        Args:
            input_text: Text to pipe to Claude
            prompt: The prompt message
            output_format: Output format ('text' or 'json')
            **kwargs: Additional arguments
            
        Returns:
            Dict containing the result and metadata
        """
        cmd = ["claude", "-p", prompt]
        
        if output_format == "json":
            cmd.extend(["--output-format", "json"])
        
        for key, value in kwargs.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        try:
            result = subprocess.run(
                cmd,
                input=input_text,
                capture_output=True,
                text=True,
                cwd=self.working_dir,
                check=False
            )
            
            if output_format == "json":
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    return {
                        "type": "error",
                        "error": "Failed to parse JSON output",
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
            else:
                return {
                    "type": "result",
                    "result": result.stdout,
                    "stderr": result.stderr
                }
                
        except Exception as e:
            return {
                "type": "error",
                "error": str(e)
            }


def simple_claude_prompt(
    message: str,
    allow_web_search: bool = False,
    output_json: bool = False
) -> str:
    """
    Simple function to call Claude CLI with a prompt.
    
    Args:
        message: The prompt message to send to Claude
        allow_web_search: Whether to allow web search permission
        output_json: Whether to request JSON output format
        
    Returns:
        String result from Claude or error message
    """
    cmd = ["claude", "-p", message]
    
    if output_json:
        cmd.extend(["--output-format", "json"])
    
    if allow_web_search:
        cmd.extend(["--allowedTools", "WebSearch"])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if output_json and result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                return json.dumps(data, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                return f"Error parsing JSON: {result.stdout}"
        
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
        
    except Exception as e:
        return f"Error executing command: {str(e)}"


if __name__ == "__main__":
    # Example 1: Simple prompt
    print("Example 1: Simple calculation")
    print("-" * 50)
    result = simple_claude_prompt("What is 2 + 2?")
    print(f"Result: {result}")
    print()
    
    # Example 2: Using the class with JSON output
    print("Example 2: JSON output with class")
    print("-" * 50)
    cli = ClaudeCLI()
    result = cli.prompt(
        "List 3 Python best practices",
        output_format="json",
        max_turns=1
    )
    if result.get("type") == "result" or "result" in result:
        print(f"Success: {result.get('result', 'See full output')}")
        if "total_cost_usd" in result:
            print(f"Cost: ${result['total_cost_usd']:.4f}")
    else:
        print(f"Error: {result}")
    print()
    
    # Example 3: Web search (requires permission)
    print("Example 3: Web search (with permission)")
    print("-" * 50)
    result = cli.prompt(
        "What is the current Bitcoin price?",
        output_format="json",
        allow=["WebSearch"],
        max_turns=3
    )
    if "result" in result:
        print(f"Result: {result['result'][:200]}...")  # Show first 200 chars
    else:
        print(f"Response: {result}")
    print()
    
    # Example 4: Piping input
    print("Example 4: Piping text for analysis")
    print("-" * 50)
    code_sample = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    result = cli.pipe_input(
        code_sample,
        "Analyze this code and suggest improvements",
        output_format="text"
    )
    print(f"Analysis: {result.get('result', 'No result')[:300]}...")