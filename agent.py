#!/usr/bin/env python3
"""Agent layer for LLM tool use.

Handles tool calls from LLM providers (OpenAI, Anthropic, etc.)
and executes tools like browser automation.
"""
import logging
import json
import asyncio
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import tool integrations
try:
    from browser_integration import browser_integration
    HAS_BROWSER_TOOLS = True
except ImportError:
    HAS_BROWSER_TOOLS = False
    logger.warning("Browser tools not available")


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


class AgentExecutor:
    """Executes tools called by LLM agents."""

    def __init__(self):
        """Initialize agent executor."""
        self.tools = {}
        self.register_tools()

    def register_tools(self):
        """Register all available tools."""
        if HAS_BROWSER_TOOLS:
            # Register browser tools
            browser_tools = browser_integration.get_tools_for_provider("openai")
            for tool in browser_tools:
                self.tools[tool["name"]] = {
                    "definition": tool,
                    "executor": self._execute_browser_tool,
                }
            logger.info(f"Registered {len(browser_tools)} browser tools")

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get all available tool definitions."""
        return [tool["definition"] for tool in self.tools.values()]

    async def _execute_browser_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> str:
        """Execute a browser tool."""
        if not HAS_BROWSER_TOOLS:
            return "Browser tools not available"

        try:
            result = await browser_integration.execute_tool(
                tool_name,
                tool_input,
                session_id,
            )
            return result
        except Exception as e:
            logger.error(f"Tool execution error: {e}", extra={"tool": tool_name})
            return f"Error executing {tool_name}: {e}"

    async def execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> str:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
            session_id: Chat session ID

        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}"

        tool_info = self.tools[tool_name]
        executor = tool_info["executor"]

        logger.info(
            f"Executing tool: {tool_name}",
            extra={"session_id": session_id},
        )

        return await executor(tool_name, tool_input, session_id)

    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute multiple tool calls and return results.

        Args:
            tool_calls: List of tool calls from LLM
            session_id: Chat session ID

        Returns:
            List of tool results
        """
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name") or tool_call.get("name")
            tool_id = tool_call.get("id")
            arguments_str = tool_call.get("function", {}).get("arguments") or tool_call.get("arguments", "{}")

            try:
                # Parse arguments
                if isinstance(arguments_str, str):
                    arguments = json.loads(arguments_str)
                else:
                    arguments = arguments_str

                # Execute tool
                result = await self.execute_tool(tool_name, arguments, session_id)

                results.append({
                    "tool_call_id": tool_id,
                    "tool_name": tool_name,
                    "result": result,
                    "status": "success",
                })
            except Exception as e:
                logger.error(f"Error in tool call {tool_id}: {e}")
                results.append({
                    "tool_call_id": tool_id,
                    "tool_name": tool_name,
                    "result": f"Error: {e}",
                    "status": "error",
                })

        return results


# Global executor instance
agent_executor = AgentExecutor()


# Playwright objects are bound to the loop that created them, so every browser
# call must run on one long-lived loop rather than a fresh loop per request.
_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name="agent-loop", daemon=True).start()
        return _loop


def run_sync(coro, timeout: float = 120) -> Any:
    """Run a coroutine on the shared agent loop from synchronous code."""
    return asyncio.run_coroutine_threadsafe(coro, _get_loop()).result(timeout)


def get_system_prompt_addon() -> str:
    """Get system prompt addition for tool use."""
    if not agent_executor.get_available_tools():
        return ""

    addon = """
## Available Tools

You have access to the following tools to help answer questions:

"""

    for tool in agent_executor.get_available_tools():
        addon += f"### {tool['name']}\n"
        addon += f"{tool.get('description', 'No description')}\n\n"

    addon += """
### Using Tools

When you need to use a tool:
1. Identify which tool would best answer the user's question
2. Call the tool with the required parameters
3. Analyze the results and provide a clear answer to the user
4. If a tool call fails, try a different approach

### Tool Guidelines

- Always open a browser before using other browser tools
- Use clear, specific CSS selectors when clicking elements
- Take screenshots to verify your actions
- Close browsers when done to free resources
- Handle errors gracefully and explain what went wrong
- Ask for clarification if parameters are unclear
"""

    return addon
