#!/usr/bin/env python3
"""Tests for browser automation tools.

Run with: python -m pytest test_browser_tools.py -v

Note: Requires playwright or selenium to be installed:
    pip install playwright && playwright install chromium
    or
    pip install selenium
"""
import pytest
import asyncio
from browser_integration import browser_integration, BrowserToolsIntegration
from browser_tools import browser_tools


class TestBrowserToolsIntegration:
    """Test browser tools integration."""

    @pytest.mark.skipif(not browser_tools.is_available(), reason="Browser tools not installed")
    @pytest.mark.asyncio
    async def test_open_browser(self):
        """Test opening a browser."""
        result = await browser_integration.execute_tool(
            "open_browser",
            {"session_id": "test_session_1"},
        )
        assert "opened successfully" in result.lower() or "open" in result.lower()

    @pytest.mark.skipif(not browser_tools.is_available(), reason="Browser tools not installed")
    @pytest.mark.asyncio
    async def test_navigate(self):
        """Test navigating to a URL."""
        # Open browser first
        await browser_integration.execute_tool(
            "open_browser",
            {"session_id": "test_session_2"},
        )

        # Navigate to allowed domain
        result = await browser_integration.execute_tool(
            "navigate",
            {"session_id": "test_session_2", "url": "python.org"},
        )

        assert "navigated" in result.lower() or "python" in result.lower()

    @pytest.mark.skipif(not browser_tools.is_available(), reason="Browser tools not installed")
    @pytest.mark.asyncio
    async def test_navigate_invalid_domain(self):
        """Test that invalid domains are blocked."""
        # Open browser first
        await browser_integration.execute_tool(
            "open_browser",
            {"session_id": "test_session_3"},
        )

        # Try to navigate to non-whitelisted domain
        result = await browser_integration.execute_tool(
            "navigate",
            {"session_id": "test_session_3", "url": "example-malicious-site.com"},
        )

        # Should be blocked
        assert "not allowed" in result.lower() or "error" in result.lower()

    @pytest.mark.skipif(not browser_tools.is_available(), reason="Browser tools not installed")
    @pytest.mark.asyncio
    async def test_take_screenshot(self):
        """Test taking a screenshot."""
        # Open browser and navigate
        await browser_integration.execute_tool(
            "open_browser",
            {"session_id": "test_session_4"},
        )
        await browser_integration.execute_tool(
            "navigate",
            {"session_id": "test_session_4", "url": "python.org"},
        )

        # Take screenshot
        result = await browser_integration.execute_tool(
            "take_screenshot",
            {
                "session_id": "test_session_4",
                "name": "test_screenshot",
            },
        )

        assert "screenshot" in result.lower() or "saved" in result.lower()

    @pytest.mark.skipif(not browser_tools.is_available(), reason="Browser tools not installed")
    @pytest.mark.asyncio
    async def test_close_browser(self):
        """Test closing a browser."""
        # Open browser
        await browser_integration.execute_tool(
            "open_browser",
            {"session_id": "test_session_5"},
        )

        # Close browser
        result = await browser_integration.execute_tool(
            "close_browser",
            {"session_id": "test_session_5"},
        )

        assert "closed" in result.lower()

    def test_get_tools_for_provider(self):
        """Test getting available tools."""
        tools = browser_integration.get_tools_for_provider("openai")

        if browser_tools.is_available():
            assert len(tools) > 0
            tool_names = [t["name"] for t in tools]
            assert "open_browser" in tool_names
            assert "navigate" in tool_names
            assert "click_element" in tool_names
        else:
            assert len(tools) == 0

    def test_system_prompt_addon(self):
        """Test system prompt addon generation."""
        prompt = browser_integration.create_system_prompt_addon()

        if browser_tools.is_available():
            assert len(prompt) > 0
            assert "browser" in prompt.lower()
            assert "automation" in prompt.lower()
        else:
            assert prompt == ""

    def test_get_session_info(self):
        """Test getting session information."""
        # Session that doesn't exist
        info = browser_integration.get_session_info("nonexistent_session")
        assert info["status"] == "not_open"


class TestBrowserToolsValidation:
    """Test browser tools validation."""

    def test_validate_url_safe(self):
        """Test that safe URLs are allowed."""
        assert browser_tools._validate_url("google.com") is True
        assert browser_tools._validate_url("https://google.com") is True
        assert browser_tools._validate_url("wikipedia.org") is True

    def test_validate_url_localhost(self):
        """Test that localhost is allowed."""
        assert browser_tools._validate_url("localhost:8000") is True
        assert browser_tools._validate_url("127.0.0.1") is True

    def test_validate_url_blocked_keywords(self):
        """Test that blocked keywords are detected."""
        assert browser_tools._validate_url("malware-site.com") is False
        assert browser_tools._validate_url("phishing.com") is False

    def test_validate_url_unsafe_domain(self):
        """Test that non-whitelisted domains are blocked."""
        # This domain is not in the allowed list
        assert browser_tools._validate_url("random-site-12345.com") is False


class TestBrowserToolsEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.skipif(not browser_tools.is_available(), reason="Browser tools not installed")
    @pytest.mark.asyncio
    async def test_double_open(self):
        """Test opening browser twice."""
        # Open first time
        result1 = await browser_integration.execute_tool(
            "open_browser",
            {"session_id": "test_double"},
        )
        assert "open" in result1.lower()

        # Try to open again
        result2 = await browser_integration.execute_tool(
            "open_browser",
            {"session_id": "test_double"},
        )
        # Should indicate browser already open
        assert "already open" in result2.lower() or "open" in result2.lower()

    @pytest.mark.skipif(not browser_tools.is_available(), reason="Browser tools not installed")
    @pytest.mark.asyncio
    async def test_tool_on_closed_browser(self):
        """Test using tools on closed browser."""
        # Try to navigate without opening browser
        result = await browser_integration.execute_tool(
            "navigate",
            {"session_id": "nonexistent"},
        )

        # Should indicate browser not open
        assert "not open" in result.lower() or "error" in result.lower()

    def test_available_check(self):
        """Test browser availability check."""
        available = browser_tools.is_available()
        # Should return boolean
        assert isinstance(available, bool)


# Integration test that simulates real usage
@pytest.mark.skipif(not browser_tools.is_available(), reason="Browser tools not installed")
@pytest.mark.asyncio
async def test_integration_search_workflow():
    """Test a complete search workflow."""
    # This is a realistic example of using the browser tools
    integration = BrowserToolsIntegration()

    # Open browser
    result = await integration.execute_tool(
        "open_browser",
        {"session_id": "search_workflow"},
    )
    assert "open" in result.lower()

    # Navigate to Google
    result = await integration.execute_tool(
        "navigate",
        {"session_id": "search_workflow", "url": "google.com"},
    )
    assert "navigate" in result.lower() or "google" in result.lower()

    # Take screenshot of homepage
    result = await integration.execute_tool(
        "take_screenshot",
        {"session_id": "search_workflow", "name": "google_homepage"},
    )
    assert "screenshot" in result.lower()

    # Check session info
    info = integration.get_session_info("search_workflow")
    assert info["status"] == "open"
    assert info["current_url"] != ""

    # Close browser
    result = await integration.execute_tool(
        "close_browser",
        {"session_id": "search_workflow"},
    )
    assert "closed" in result.lower()


# Example of how to run tests
if __name__ == "__main__":
    print("Run tests with: pytest test_browser_tools.py -v")
    print("\nTo enable browser tools:")
    print("  pip install playwright && playwright install")
    print("  or")
    print("  pip install selenium")
