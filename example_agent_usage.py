#!/usr/bin/env python3
"""Example: Using browser tools agent with different LLM providers.

This file demonstrates how to use the agent system with:
- OpenAI
- Anthropic (Claude)
- Generic OpenAI-compatible APIs
"""

import asyncio
import json
from typing import Optional

# Note: These imports assume the necessary libraries are installed
# Install with: pip install openai anthropic playwright


# ============================================================================
# Example 1: Using with OpenAI
# ============================================================================

async def example_openai():
    """Example using OpenAI GPT-4 with browser tools."""
    try:
        from openai import OpenAI
        from agent import agent_executor
    except ImportError:
        print("Required: pip install openai playwright")
        return

    client = OpenAI(api_key="sk-your-key-here")

    # Get browser tools
    tools = agent_executor.get_available_tools()
    if not tools:
        print("No tools available. Install: pip install playwright && playwright install")
        return

    print(f"Available tools: {[t['name'] for t in tools]}\n")

    messages = [
        {
            "role": "user",
            "content": "Search Google for 'Python programming tutorials' and tell me what you find"
        }
    ]

    print("OpenAI: Sending request with browser tools...\n")

    # Initial request
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    # Process tool calls if any
    iteration = 0
    max_iterations = 10  # Prevent infinite loops

    while response.choices[0].finish_reason == "tool_calls" and iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}: Tool calls detected")

        tool_calls = response.choices[0].message.tool_calls

        for tool_call in tool_calls:
            print(f"  - Executing: {tool_call.function.name}")

        # Add assistant message
        messages.append(response.choices[0].message)

        # Execute tools
        tool_results = await agent_executor.execute_tool_calls(
            [
                {
                    "id": call.id,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                }
                for call in tool_calls
            ],
            session_id="example_openai",
        )

        # Add tool results
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": result["result"]
            })
            print(f"  ✓ {result['tool_name']}: {result['status']}")

        # Continue conversation
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            tools=tools,
        )

    # Get final answer
    final_answer = response.choices[0].message.content
    print(f"\nFinal Answer:\n{final_answer}\n")


# ============================================================================
# Example 2: Using with Anthropic (Claude)
# ============================================================================

async def example_anthropic():
    """Example using Anthropic Claude with browser tools."""
    try:
        from anthropic import Anthropic
        from agent import agent_executor
    except ImportError:
        print("Required: pip install anthropic playwright")
        return

    client = Anthropic(api_key="sk-ant-your-key-here")

    # Get browser tools
    tools = agent_executor.get_available_tools()
    if not tools:
        print("No tools available. Install: pip install playwright && playwright install")
        return

    print(f"Available tools: {[t['name'] for t in tools]}\n")

    messages = [
        {
            "role": "user",
            "content": "Open a browser and check if python.org is working"
        }
    ]

    print("Anthropic: Sending request with browser tools...\n")

    # Initial request
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        tools=tools,
        messages=messages,
    )

    # Process tool uses if any
    iteration = 0
    max_iterations = 10

    while response.stop_reason == "tool_use" and iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}: Tool use detected")

        # Extract tool uses
        tool_uses = [block for block in response.content if block.type == "tool_use"]

        for tool_use in tool_uses:
            print(f"  - Executing: {tool_use.name}")

        # Execute tools
        tool_results = await agent_executor.execute_tool_calls(
            [
                {
                    "id": use.id,
                    "name": use.name,
                    "arguments": use.input,
                }
                for use in tool_uses
            ],
            session_id="example_anthropic",
        )

        # Build next message with results
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": result["tool_call_id"],
                    "content": result["result"],
                }
                for result in tool_results
            ]
        })

        for result in tool_results:
            print(f"  ✓ {result['tool_name']}: {result['status']}")

        # Continue conversation
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            tools=tools,
            messages=messages,
        )

    # Get final text response
    text_blocks = [block for block in response.content if block.type == "text"]
    final_answer = "".join([block.text for block in text_blocks])
    print(f"\nFinal Answer:\n{final_answer}\n")


# ============================================================================
# Example 3: Direct Agent Usage
# ============================================================================

async def example_direct_agent():
    """Example of using agent directly without LLM provider."""
    from agent import agent_executor

    print("Direct Agent Usage Example\n")
    print("Available tools:")
    tools = agent_executor.get_available_tools()
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")

    print("\n" + "="*60)
    print("Simulating tool calls...\n")

    # Manually call tools (as if LLM requested them)
    tool_calls = [
        {
            "id": "call_1",
            "name": "open_browser",
            "arguments": {"session_id": "demo_1"}
        },
        {
            "id": "call_2",
            "name": "navigate",
            "arguments": {"session_id": "demo_1", "url": "python.org"}
        },
        {
            "id": "call_3",
            "name": "take_screenshot",
            "arguments": {"session_id": "demo_1", "name": "python_site"}
        },
        {
            "id": "call_4",
            "name": "close_browser",
            "arguments": {"session_id": "demo_1"}
        }
    ]

    results = await agent_executor.execute_tool_calls(tool_calls, session_id="demo")

    print("Tool Results:")
    for result in results:
        status_emoji = "✓" if result["status"] == "success" else "✗"
        print(f"{status_emoji} {result['tool_name']}: {result['result'][:80]}...")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all examples."""
    print("=" * 70)
    print("AGENT TOOL USE EXAMPLES")
    print("=" * 70 + "\n")

    # Example 1: Direct agent usage (no API key needed)
    print("Example 1: Direct Agent Usage")
    print("-" * 70)
    try:
        await example_direct_agent()
    except Exception as e:
        print(f"Error: {e}\n")

    print("\n" + "=" * 70 + "\n")

    # Example 2: OpenAI (requires API key)
    print("Example 2: OpenAI with Tool Use")
    print("-" * 70)
    print("To run this, set your OpenAI key in example_openai() function")
    print("Skipping for now...\n")
    # await example_openai()

    print("\n" + "=" * 70 + "\n")

    # Example 3: Anthropic (requires API key)
    print("Example 3: Anthropic with Tool Use")
    print("-" * 70)
    print("To run this, set your Anthropic key in example_anthropic() function")
    print("Skipping for now...\n")
    # await example_anthropic()


if __name__ == "__main__":
    print("\nAgent Implementation Examples")
    print("=============================\n")

    print("Setup (if you haven't already):")
    print("  pip install playwright")
    print("  playwright install chromium")
    print("\nOr with Selenium:")
    print("  pip install selenium\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
