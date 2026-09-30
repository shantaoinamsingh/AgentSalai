#!/usr/bin/env python3
"""Browser automation tools for Salai chatbot.

Provides capabilities for the LLM to:
- Open and navigate web pages
- Extract text content from pages
- Click buttons and interact with elements
- Fill forms
- Take screenshots
- Execute JavaScript

Security Note:
- Sanitize all user input before passing to browser
- Limit allowed domains to prevent abuse
- Add rate limiting for browser operations
- Log all browser activities
"""
import json
import logging
import os
import re
import threading
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import base64
from urllib.parse import parse_qs, quote_plus, urlparse

logger = logging.getLogger(__name__)

# Starts as ALLOWED_DOMAINS; "Always allow" in the chat adds to it.
TRUSTED_DOMAINS_FILE = os.getenv(
    "BROWSER_TRUSTED_DOMAINS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "trusted_domains.json"),
)

# Try to import Playwright first (preferred), fall back to Selenium
try:
    from playwright.async_api import async_playwright, Browser, Page
    HAS_PLAYWRIGHT = True
    BROWSER_ENGINE = "playwright"
except ImportError:
    HAS_PLAYWRIGHT = False
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.options import Options
        HAS_SELENIUM = True
        BROWSER_ENGINE = "selenium"
    except ImportError:
        HAS_SELENIUM = False
        BROWSER_ENGINE = None


@dataclass
class BrowserToolError(Exception):
    """Error from browser tools."""
    pass


@dataclass
class BrowserContext:
    """Context for browser operations."""
    session_id: str
    browser: Optional[Any] = None
    page: Optional[Any] = None
    current_url: str = ""
    page_content: str = ""
    screenshots: Dict[str, bytes] = None

    def __post_init__(self):
        if self.screenshots is None:
            self.screenshots = {}


class BrowserToolManager:
    """Manages browser automation for the chatbot."""

    # Initial trusted list. Other domains are not blocked: the user is asked first.
    ALLOWED_DOMAINS = [
        "example.com",
        "facebook.com",
        "google.com",
        "duckduckgo.com",
        "bing.com",
        "wikipedia.org",
        "github.com",
        "stackoverflow.com",
        "python.org",
        "nodejs.org",
        "docker.com",
        "openai.com",
        "anthropic.com",
        "dentsu.com",
        "nike.in",
        "addidas.com",
        "linkedin.com",
    ]

    # Blocked keywords in URLs
    BLOCKED_KEYWORDS = [
        "malware",
        "phishing",
        "exploit",
        "ransomware",
        "sql-injection",
    ]

    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    PAGE_LOAD_TIMEOUT = 30  # seconds
    MAX_SCREENSHOTS = 10
    MAX_PAGE_SIZE = 1_000_000  # 1MB limit

    def __init__(self):
        """Initialize browser tool manager."""
        self.contexts: Dict[str, BrowserContext] = {}
        self.engine = BROWSER_ENGINE
        self._trust_lock = threading.Lock()
        self.trusted = self._load_trusted()
        self.session_approvals: Dict[str, set] = {}

        if not HAS_PLAYWRIGHT and not HAS_SELENIUM:
            logger.warning(
                "No browser automation library found. "
                "Install with: pip install playwright selenium"
            )

    def is_available(self) -> bool:
        """Check if browser tools are available."""
        return bool(self.engine)

    @staticmethod
    def domain_of(url: str) -> str:
        # "host:port" parses as a scheme, so test for "://" instead.
        if "://" not in url:
            url = f"https://{url}"
        try:
            return (urlparse(url).hostname or "").lower()
        except ValueError:
            return ""

    def is_safe_url(self, url: str) -> bool:
        """Hard safety checks that no user approval can override."""
        if any(keyword in url.lower() for keyword in self.BLOCKED_KEYWORDS):
            return False
        if "://" in url and not url.lower().startswith(("http://", "https://")):
            return False
        return bool(self.domain_of(url))

    def _load_trusted(self) -> set:
        try:
            with open(TRUSTED_DOMAINS_FILE, encoding="utf-8") as f:
                return {d.lower() for d in json.load(f)}
        except FileNotFoundError:
            return set(self.ALLOWED_DOMAINS)
        except (OSError, ValueError) as e:
            logger.error(f"Could not read {TRUSTED_DOMAINS_FILE}: {e}; using defaults")
            return set(self.ALLOWED_DOMAINS)

    def trusted_domains(self) -> List[str]:
        return sorted(self.trusted)

    def trust_domain(self, domain: str) -> None:
        """Trust a domain for every chat, persisted across restarts."""
        with self._trust_lock:
            self.trusted.add(domain.lower())
            with open(TRUSTED_DOMAINS_FILE, "w", encoding="utf-8") as f:
                json.dump(sorted(self.trusted), f, indent=2)
        logger.info(f"Domain trusted permanently: {domain}")

    def approve_domain(self, session_id: str, domain: str) -> None:
        """Allow a domain for one browser session only."""
        self.session_approvals.setdefault(session_id, set()).add(domain.lower())

    def is_trusted(self, url: str, session_id: Optional[str] = None) -> bool:
        domain = self.domain_of(url)
        if domain in ("localhost", "127.0.0.1", "0.0.0.0"):
            return True
        allowed = self.trusted | self.session_approvals.get(session_id, set())
        return any(domain == d or domain.endswith("." + d) for d in allowed)

    def _validate_url(self, url: str, session_id: Optional[str] = None) -> bool:
        """True when the URL is safe and trusted (or approved for this session)."""
        return self.is_safe_url(url) and self.is_trusted(url, session_id)

    async def open_browser(self, session_id: str, headless: bool = True) -> str:
        """Open a new browser instance.

        Args:
            session_id: Unique session identifier
            headless: Run browser in headless mode

        Returns:
            Status message
        """
        if not self.is_available():
            return "Browser tools not available. Install playwright or selenium."

        if session_id in self.contexts:
            return f"Browser already open for session {session_id}"

        try:
            if self.engine == "playwright":
                return await self._open_playwright(session_id, headless)
            elif self.engine == "selenium":
                return self._open_selenium(session_id, headless)
        except Exception as e:
            logger.error(f"Error opening browser: {e}")
            raise BrowserToolError(f"Failed to open browser: {e}")

    async def _open_playwright(self, session_id: str, headless: bool) -> str:
        """Open browser using Playwright."""
        try:
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(headless=headless)
            page = await browser.new_page()

            self.contexts[session_id] = BrowserContext(
                session_id=session_id,
                browser=browser,
                page=page,
            )

            logger.info(f"Opened Playwright browser for session {session_id}")
            return f"Browser opened successfully (Playwright, headless={headless})"
        except Exception as e:
            raise BrowserToolError(f"Playwright error: {e}")

    def _open_selenium(self, session_id: str, headless: bool) -> str:
        """Open browser using Selenium."""
        try:
            options = Options()
            if headless:
                options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")

            driver = webdriver.Chrome(options=options)

            self.contexts[session_id] = BrowserContext(
                session_id=session_id,
                browser=driver,
                page=driver,
            )

            logger.info(f"Opened Selenium browser for session {session_id}")
            return f"Browser opened successfully (Selenium, headless={headless})"
        except Exception as e:
            raise BrowserToolError(f"Selenium error: {e}")

    async def navigate(self, session_id: str, url: str) -> str:
        """Navigate to a URL.

        Args:
            session_id: Browser session ID
            url: URL to navigate to

        Returns:
            Status message with page info
        """
        if session_id not in self.contexts:
            return "Browser not open. Use open_browser first."

        if not self.is_safe_url(url):
            return f"URL not allowed (blocked as unsafe): {url}"
        if not self.is_trusted(url, session_id):
            return (
                f"URL not allowed yet: {self.domain_of(url)} is not a trusted domain "
                "and needs the user's approval."
            )

        # Ensure URL has scheme
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        context = self.contexts[session_id]
        retries = 0

        while retries < self.MAX_RETRIES:
            try:
                if self.engine == "playwright":
                    await context.page.goto(
                        url, wait_until="networkidle", timeout=self.PAGE_LOAD_TIMEOUT * 1000
                    )
                else:  # selenium
                    context.page.get(url)
                    WebDriverWait(context.page, self.PAGE_LOAD_TIMEOUT).until(
                        EC.presence_of_all_elements_located((By.TAG_NAME, "body"))
                    )

                context.current_url = url
                context.page_content = await self._get_page_content(session_id)

                return f"✓ Navigated to {url}\n\nPage content preview:\n{context.page_content[:3000]}..."
            except Exception as e:
                retries += 1
                if retries < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY)
                else:
                    logger.error(f"Navigation error after {self.MAX_RETRIES} retries: {e}")
                    return f"Failed to navigate to {url}: {e}"

    async def _get_page_content(self, session_id: str) -> str:
        """Extract text content from current page."""
        context = self.contexts[session_id]

        try:
            if self.engine == "playwright":
                content = await context.page.inner_text("body")
            else:  # selenium
                content = context.page.find_element(By.TAG_NAME, "body").text

            # Limit size
            if len(content) > self.MAX_PAGE_SIZE:
                content = content[:self.MAX_PAGE_SIZE] + "\n... (content truncated)"

            return content
        except Exception as e:
            logger.warning(f"Error extracting page content: {e}")
            return "(Unable to extract page content)"

    async def click_element(self, session_id: str, selector: str) -> str:
        """Click an element on the page.

        Args:
            session_id: Browser session ID
            selector: CSS selector for element

        Returns:
            Status message
        """
        if session_id not in self.contexts:
            return "Browser not open."

        context = self.contexts[session_id]

        try:
            if self.engine == "playwright":
                await context.page.click(selector, timeout=5000)
            else:  # selenium
                element = WebDriverWait(context.page, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                element.click()

            # Wait a moment and get updated content
            await self._wait_for_navigation(session_id)
            context.page_content = await self._get_page_content(session_id)

            return f"✓ Clicked element: {selector}"
        except Exception as e:
            logger.error(f"Click error: {e}")
            return f"Failed to click element: {e}"

    async def fill_input(self, session_id: str, selector: str, text: str) -> str:
        """Fill a text input field.

        Args:
            session_id: Browser session ID
            selector: CSS selector for input
            text: Text to enter

        Returns:
            Status message
        """
        if session_id not in self.contexts:
            return "Browser not open."

        context = self.contexts[session_id]

        try:
            if self.engine == "playwright":
                await context.page.fill(selector, text)
            else:  # selenium
                element = WebDriverWait(context.page, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                element.clear()
                element.send_keys(text)

            return f"✓ Filled input: {selector} = {text[:50]}"
        except Exception as e:
            logger.error(f"Fill error: {e}")
            return f"Failed to fill input: {e}"

    async def take_screenshot(self, session_id: str, name: str = None) -> str:
        """Take a screenshot of the current page.

        Args:
            session_id: Browser session ID
            name: Optional name for screenshot

        Returns:
            Status message with screenshot info
        """
        if session_id not in self.contexts:
            return "Browser not open."

        context = self.contexts[session_id]

        if len(context.screenshots) >= self.MAX_SCREENSHOTS:
            return f"Maximum screenshots ({self.MAX_SCREENSHOTS}) reached"

        name = name or f"screenshot_{len(context.screenshots) + 1}"

        try:
            if self.engine == "playwright":
                screenshot = await context.page.screenshot()
            else:  # selenium
                screenshot = context.page.get_screenshot_as_png()

            context.screenshots[name] = screenshot

            return f"✓ Screenshot saved: {name} ({len(screenshot)} bytes)"
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            return f"Failed to take screenshot: {e}"

    async def execute_js(self, session_id: str, script: str) -> str:
        """Execute JavaScript on the page.

        Args:
            session_id: Browser session ID
            script: JavaScript code to execute

        Returns:
            Result of JavaScript execution
        """
        if session_id not in self.contexts:
            return "Browser not open."

        context = self.contexts[session_id]

        # Basic security check
        dangerous_keywords = ["localStorage", "sessionStorage", "document.cookie"]
        if any(keyword in script for keyword in dangerous_keywords):
            return "Script contains blocked keywords"

        try:
            if self.engine == "playwright":
                result = await context.page.evaluate(script)
            else:  # selenium
                result = context.page.execute_script(script)

            return f"✓ JavaScript executed: {str(result)[:200]}"
        except Exception as e:
            logger.error(f"JavaScript error: {e}")
            return f"JavaScript error: {e}"

    async def _wait_for_navigation(self, session_id: str, timeout: int = 5000):
        """Wait for page to load after an action."""
        if session_id not in self.contexts:
            return

        context = self.contexts[session_id]

        try:
            if self.engine == "playwright":
                await context.page.wait_for_load_state("networkidle", timeout=timeout)
            else:
                time.sleep(1)  # Basic wait for Selenium
        except Exception:
            pass  # Timeout is acceptable

    async def close_browser(self, session_id: str) -> str:
        """Close a browser instance.

        Args:
            session_id: Browser session ID

        Returns:
            Status message
        """
        if session_id not in self.contexts:
            return "Browser not open."

        context = self.contexts[session_id]

        try:
            if self.engine == "playwright":
                await context.browser.close()
            else:  # selenium
                context.browser.quit()

            del self.contexts[session_id]
            logger.info(f"Closed browser for session {session_id}")
            return f"Browser closed. Captured {len(context.screenshots)} screenshots."
        except Exception as e:
            logger.error(f"Close error: {e}")
            return f"Error closing browser: {e}"

    async def web_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Top Bing results as {title, url, snippet}, in a throwaway headless browser.

        Needs no LLM, so it still works when the model provider is down.
        """
        if self.engine != "playwright":
            raise BrowserToolError("Web search needs Playwright (pip install playwright)")
        url = f"https://www.bing.com/search?q={quote_plus(query)}"
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                # Bing now and then serves a page with no results, a half-rendered
                # one, or reloads itself (&rdr=1) mid-read. A fresh request fixes all three.
                for _ in range(3):
                    try:
                        await page.goto(url, wait_until="domcontentloaded", timeout=self.PAGE_LOAD_TIMEOUT * 1000)
                        await page.wait_for_selector("li.b_algo", timeout=8_000)
                        items = await page.eval_on_selector_all(
                            "li.b_algo",
                            """els => els.map(e => ({
                                title: e.querySelector('h2')?.innerText || '',
                                url: e.querySelector('h2 a')?.href || '',
                                snippet: (e.querySelector('.b_caption p') || e.querySelector('p'))?.innerText || ''
                            }))""",
                        )
                        results = _parse_search_items(items, max_results)
                        if results:
                            return results
                        logger.info("Bing results page had no readable results; retrying")
                    except Exception as e:
                        logger.info(
                            f"Bing results not readable yet ({e.__class__.__name__}) "
                            f"at {page.url[:120]!r}; retrying"
                        )
                return []
            finally:
                await browser.close()

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available browser tools for the LLM."""
        if not self.is_available():
            return []

        return [
            {
                "name": "open_browser",
                "description": "Open a web browser instance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Unique session identifier",
                        },
                        "headless": {
                            "type": "boolean",
                            "description": "Run in headless mode (no GUI)",
                            "default": True,
                        },
                    },
                    "required": ["session_id"],
                },
            },
            {
                "name": "navigate",
                "description": "Navigate to a URL in the browser",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Browser session ID",
                        },
                        "url": {
                            "type": "string",
                            "description": "URL to navigate to (must be whitelisted)",
                        },
                    },
                    "required": ["session_id", "url"],
                },
            },
            {
                "name": "click_element",
                "description": "Click an element on the page using CSS selector",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Browser session ID",
                        },
                        "selector": {
                            "type": "string",
                            "description": "CSS selector (e.g., '#button-id', '.btn-primary')",
                        },
                    },
                    "required": ["session_id", "selector"],
                },
            },
            {
                "name": "fill_input",
                "description": "Fill a text input field",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Browser session ID",
                        },
                        "selector": {
                            "type": "string",
                            "description": "CSS selector for input field",
                        },
                        "text": {
                            "type": "string",
                            "description": "Text to fill in",
                        },
                    },
                    "required": ["session_id", "selector", "text"],
                },
            },
            {
                "name": "take_screenshot",
                "description": "Take a screenshot of the current page",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Browser session ID",
                        },
                        "name": {
                            "type": "string",
                            "description": "Optional name for the screenshot",
                        },
                    },
                    "required": ["session_id"],
                },
            },
            {
                "name": "execute_js",
                "description": "Execute JavaScript on the page",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Browser session ID",
                        },
                        "script": {
                            "type": "string",
                            "description": "JavaScript code to execute",
                        },
                    },
                    "required": ["session_id", "script"],
                },
            },
            {
                "name": "close_browser",
                "description": "Close the browser instance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Browser session ID",
                        },
                    },
                    "required": ["session_id"],
                },
            },
        ]


# Global instance
def _parse_search_items(items: List[Dict[str, str]], max_results: int) -> List[Dict[str, str]]:
    results = []
    for item in items:
        link = _unwrap_bing_link(item.get("url", ""))
        title = " ".join(item.get("title", "").split())
        if title and link.startswith(("http://", "https://")):
            results.append({"title": title, "url": link, "snippet": " ".join(item.get("snippet", "").split())})
        if len(results) >= max_results:
            break
    return results


def _unwrap_bing_link(href: str) -> str:
    """Bing result links are click-tracking redirects; the target is base64 in `u=a1...`."""
    parsed = urlparse(href)
    if not parsed.netloc.endswith("bing.com") or not parsed.path.startswith("/ck/"):
        return href
    encoded = parse_qs(parsed.query).get("u", [""])[0]
    if not encoded.startswith("a1"):
        return href
    encoded = encoded[2:]
    try:
        return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return href


browser_tools = BrowserToolManager()
