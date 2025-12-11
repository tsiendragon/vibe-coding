#!/usr/bin/env python3
"""
Simple demonstration of Claude CLI Python wrapper
"""

from claude_cli_wrapper import simple_claude_prompt, ClaudeCLI
import json

def main():
    print("🤖 Claude CLI Python Wrapper Demo")
    print("=" * 40)
    
    # Demo 1: Simple calculation
    print("\n1️⃣ Simple Calculation:")
    result = simple_claude_prompt("Calculate 15 * 23 + 7")
    print(f"   Result: {result.strip()}")
    
    # Demo 2: Code explanation
    print("\n2️⃣ Code Explanation:")
    code_sample = "lambda x: x**2 if x > 0 else 0"
    result = simple_claude_prompt(f"Explain this Python code: {code_sample}")
    print(f"   Explanation: {result.strip()[:150]}...")
    
    # Demo 3: JSON output with metadata
    print("\n3️⃣ JSON Output with Metadata:")
    cli = ClaudeCLI()
    result = cli.prompt("What is REST API?", output_format="json", max_turns=1)
    
    if "result" in result:
        print(f"   ✅ Success!")
        print(f"   💰 Cost: ${result.get('total_cost_usd', 0):.4f}")
        print(f"   ⏱️  Duration: {result.get('duration_ms', 0)}ms")
        print(f"   📝 Response: {result['result'][:100]}...")
    else:
        print(f"   ❌ Error: {result}")
    
    # Demo 4: Web search (with permission)
    print("\n4️⃣ Web Search (requires permission):")
    try:
        result = simple_claude_prompt(
            "What's the current Bitcoin price?", 
            allow_web_search=True,
            output_json=True
        )
        data = json.loads(result)
        if "result" in data:
            print(f"   ✅ Web search successful!")
            print(f"   📊 Result: {data['result'][:100]}...")
        else:
            print(f"   ⚠️  Permission issue or error")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Demo 5: Piping content
    print("\n5️⃣ Piping Content Analysis:")
    text_content = """
    The quick brown fox jumps over the lazy dog.
    This sentence contains every letter of the alphabet.
    """
    
    result = cli.pipe_input(
        text_content,
        "Analyze this text and tell me what's special about it",
        output_format="text"
    )
    
    if "result" in result:
        print(f"   📝 Analysis: {result['result'][:150]}...")
    else:
        print(f"   ❌ Error: {result}")
    
    print("\n✨ Demo completed!")
    print("\nUsage examples:")
    print("- from claude_cli_wrapper import simple_claude_prompt")
    print("- result = simple_claude_prompt('Your question here')")
    print("- For advanced usage, see claude_cli_examples.py")

if __name__ == "__main__":
    main()