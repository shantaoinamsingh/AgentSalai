#!/usr/bin/env python3
"""Integration of browser tools with the Salai chatbot.

Allows the LLM to use browser automation tools through tool calls.
"""
import logging
import asyncio
from typing import Dict, Any, Optional
from browser_tools import browser_tools, BrowserToolError

logger = logging.getLogger(__name__)


class BrowserToolsIntegration:
    """Integration layer for browser tools with the chatbot."""

    def __init__(self):
        """Initialize browser tools integration."""
        self.tools = browser_tools
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    def get_tools_for_provider(self, provider_id: str) -> list:
        """Get browser tools for a specific provider.

        Browser tools can be used with any provider (OpenAI, Anthropic, etc.)
        that supports function calling.

        Args:
            provider_id: LLM provider ID

        Returns:
            List of tool definitions
        """
        if not self.tools.is_available():
            logger.warning("Browser tools not available - install playwright or selenium")
            return []

        return self.tools.get_available_tools()

    async def execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> str:
        """Execute a browser tool.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
            session_id: Chat session ID (for logging)

        Returns:
            Tool execution result as string
        """
        try:
            if tool_name == "open_browser":
                return await self.tools.open_browser(
                    tool_input["session_id"],
                    tool_input.get("headless", True),
                )

            elif tool_name == "navigate":
                return await self.tools.navigate(
                    tool_input["session_id"],
                    tool_input["url"],
                )

            elif tool_name == "click_element":
                return await self.tools.click_element(
                    tool_input["session_id"],
                    tool_input["selector"],
                )

            elif tool_name == "fill_input":
                return await self.tools.fill_input(
                    tool_input["session_id"],
                    tool_input["selector"],
                    tool_input["text"],
                )

            elif tool_name == "take_screenshot":
                return await self.tools.take_screenshot(
                    tool_input["session_id"],
                    tool_input.get("name"),
                )

            elif tool_name == "execute_js":
                return await self.tools.execute_js(
                    tool_input["session_id"],
                    tool_input["script"],
                )

            elif tool_name == "close_browser":
                return await self.tools.close_browser(
                    tool_input["session_id"],
                )

            else:
                return f"Unknown tool: {tool_name}"

        except BrowserToolError as e:
            logger.error(f"Browser tool error: {e}", extra={"session_id": session_id})
            return f"Error: {e}"
        except Exception as e:
            logger.error(
                f"Unexpected error in tool {tool_name}: {e}",
                extra={"session_id": session_id},
            )
            return f"Unexpected error: {e}"

    def create_system_prompt_addon(self) -> str:
        """Create a system prompt addon for browser capabilities.

        Returns:
            System prompt text describing browser tools
        """
        if not self.tools.is_available():
            return ""

        return """
## Browser Automation Capabilities

You have access to browser automation tools that allow you to:
- Open a web browser and navigate to websites
- Click buttons and links
- Fill in text inputs and forms
- Take screenshots to see what's on screen
- Execute JavaScript to interact with pages

### Browser Tool Examples

1. **Search for information**:
   - Open browser → navigate to search engine → fill search box → click search button

2. **Extract data from web pages**:
   - Navigate to URL → take screenshot → read page content

3. **Fill and submit forms**:
   - Open page → fill input fields → click submit button

4. **Interact with web applications**:
   - Navigate → click buttons → execute JavaScript → take screenshot

### Important Guidelines

- Always open a browser session before using other browser tools
- Use descriptive CSS selectors (id, class, or tag names)
- Take screenshots to verify your actions
- Close the browser when done to free resources
- Ask for clarification if a selector is not clear
- Handle errors gracefully (retry if needed)

### Security Notice

- Only allowed domains can be visited (whitelist enforced)
- Certain dangerous JavaScript operations are blocked
- All browser activities are logged for security

### Example Conversation

User: "Search for Python tutorial on Google"
Assistant:
1. I'll help you search for a Python tutorial. Let me open a browser and navigate to Google.
2. *opens browser* → *navigates to google.com* → *fills search box with "Python tutorial"* → *clicks search*
3. I found several results. Here are the top tutorials...
"""

    async def cleanup_session(self, session_id: str) -> None:
        """Clean up browser resources for a session.

        Args:
            session_id: Browser session ID
        """
        # Close all open browsers for this session
        if session_id in self.tools.contexts:
            await self.tools.close_browser(session_id)
            logger.info(f"Cleaned up browser resources for session {session_id}")

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about an active browser session.

        Args:
            session_id: Browser session ID

        Returns:
            Session information dict
        """
        if session_id not in self.tools.contexts:
            return {"status": "not_open"}

        context = self.tools.contexts[session_id]
        return {
            "status": "open",
            "current_url": context.current_url,
            "screenshots": len(context.screenshots),
            "content_length": len(context.page_content),
        }


# Global integration instance
browser_integration = BrowserToolsIntegration()


# Example usage functions
async def demo_browser_usage():
    """Demo showing how to use browser tools programmatically."""
    integration = BrowserToolsIntegration()

    # Open browser
    result = await integration.execute_tool(
        "open_browser",
        {"session_id": "demo_1"},
    )
    print(f"Open browser: {result}")

    # Navigate
    result = await integration.execute_tool(
        "navigate",
        {"session_id": "demo_1", "url": "python.org"},
    )
    print(f"Navigate: {result[:100]}...")

    # Take screenshot
    result = await integration.execute_tool(
        "take_screenshot",
        {"session_id": "demo_1", "name": "python_homepage"},
    )
    print(f"Screenshot: {result}")

    # Get session info
    info = integration.get_session_info("demo_1")
    print(f"Session info: {info}")

    # Close browser
    result = await integration.execute_tool(
        "close_browser",
        {"session_id": "demo_1"},
    )
    print(f"Close: {result}")


if __name__ == "__main__":
    # Test browser tools
    try:
        asyncio.run(demo_browser_usage())
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use browser tools, install:")
        print("  pip install playwright")
        print("  playwright install")
        print("\nOr:")
        print("  pip install selenium")
