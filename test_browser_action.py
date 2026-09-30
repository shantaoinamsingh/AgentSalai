#!/usr/bin/env python3
"""Test executing a browser action."""
import asyncio
from agent import agent_executor

async def execute_action():
    session_id = "user_session_001"
    action_name = "open_browser"
    params = {"session_id": session_id}

    print("=" * 70)
    print("BROWSER ACTION EXECUTION")
    print("=" * 70)
    print(f"Action: {action_name}")
    print(f"Session: {session_id}\n")

    result = await agent_executor.execute_tool(action_name, params)

    print(f"Result:\n{result}")
    print("\n" + "=" * 70)
    print("✓ Browser opened and ready for commands!")
    print("=" * 70)

    return result

if __name__ == "__main__":
    asyncio.run(execute_action())
