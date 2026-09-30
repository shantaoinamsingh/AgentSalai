# Browser Automation Tools for Salai

Give your Salai chatbot the ability to browse the web and interact with websites on behalf of users.

## What Can Your Chatbot Do?

With browser tools enabled, your chatbot can:

✅ **Browse Websites**
- Open browsers and navigate to URLs
- Follow links and explore websites
- Handle multiple pages and tabs

✅ **Interact with Pages**
- Click buttons and links
- Fill out forms and text inputs
- Submit data to websites
- Execute JavaScript for advanced interactions

✅ **See and Verify**
- Take screenshots to see what's on screen
- Extract text and data from pages
- Read page content and structure

✅ **Complete Tasks**
- Search for information online
- Extract structured data from websites
- Fill and submit forms
- Monitor website status
- Scrape data (within ethical bounds)

## Example Conversations

### Search for Information
```
User: "Search for the latest Python 3.13 release notes"
Chatbot: I'll search for that information for you.
[Opens browser] → [Navigates to Google] → [Searches] → [Extracts results]
Here are the latest Python 3.13 release notes...
```

### Extract Data
```
User: "Get all the links from the GitHub Python repository"
Chatbot: I'll extract the links for you.
[Navigates to repo] → [Executes JavaScript] → [Collects all links]
Found 156 links including:
- Issues page: https://...
- Pull Requests: https://...
- Documentation: https://...
```

### Complete Forms
```
User: "Sign me up for the newsletter with my email"
Chatbot: I'll fill out the newsletter signup form.
[Navigates to site] → [Fills email field] → [Clicks subscribe]
✓ Successfully subscribed to the newsletter!
```

## Installation

### Step 1: Install Browser Library

Choose one (Playwright recommended):

**Option A: Playwright (Recommended)**
```bash
pip install playwright
playwright install chromium
```

**Option B: Selenium**
```bash
pip install selenium
# Download ChromeDriver from https://chromedriver.chromium.org/
```

### Step 2: Restart Salai

```bash
python app.py
```

Browser tools are automatically detected and enabled if a library is installed.

## How to Use

### In Chat

Simply tell the chatbot what you want it to do:

```
"Search Google for Python tutorials"
"Extract all links from wikipedia.org"
"Fill out this form with my data"
"Check if the website is working"
"Get today's weather from weather.com"
```

### In Code

```python
from browser_integration import browser_integration
import asyncio

async def demo():
    # Execute browser tool
    result = await browser_integration.execute_tool(
        "open_browser",
        {"session_id": "user_123"}
    )
    print(result)  # "Browser opened successfully"
    
    # Navigate
    result = await browser_integration.execute_tool(
        "navigate",
        {"session_id": "user_123", "url": "google.com"}
    )
    print(result)  # Page content preview
    
    # Close
    result = await browser_integration.execute_tool(
        "close_browser",
        {"session_id": "user_123"}
    )

asyncio.run(demo())
```

## Available Tools

| Tool | Purpose | Example |
|------|---------|---------|
| `open_browser` | Start a browser | Open a new browser instance |
| `navigate` | Go to a URL | Visit google.com |
| `click_element` | Click buttons/links | Click search button |
| `fill_input` | Fill text fields | Enter search query |
| `take_screenshot` | Capture page | See what's on screen |
| `execute_js` | Run JavaScript | Extract page data |
| `close_browser` | End session | Clean up resources |

**Full documentation**: See [BROWSER_TOOLS.md](BROWSER_TOOLS.md)

## Security

### ✅ What's Protected

- **Domain Whitelist**: Only approved domains can be visited
- **Safe by Default**: Dangerous JavaScript blocked
- **URL Validation**: Malicious URLs rejected
- **Activity Logging**: All actions logged for auditing
- **Resource Limits**: Prevents abuse

### Default Allowed Domains
```
google.com          python.org          github.com
wikipedia.org       stackoverflow.com   docker.com
openai.com          anthropic.com       nodejs.org
```

### Add Custom Domains

Edit `browser_tools.py`:

```python
ALLOWED_DOMAINS = [
    "example.com",      # Your custom domain
    "company.com",      # Another domain
    "api.service.com",  # Subdomain support
]
```

## Configuration

### Browser Options

Control browser behavior in your code:

```python
# Use non-headless browser (shows GUI)
await browser_integration.execute_tool(
    "open_browser",
    {"session_id": "demo", "headless": false}
)

# Set longer page load timeout
browser_tools.PAGE_LOAD_TIMEOUT = 60  # seconds
```

### Resource Limits

Adjust in `browser_tools.py`:

```python
MAX_SCREENSHOTS = 10         # Per session
MAX_PAGE_SIZE = 1_000_000    # 1MB limit
PAGE_LOAD_TIMEOUT = 30       # 30 seconds
MAX_RETRIES = 3              # Retry failed navigations
```

## Troubleshooting

### "Browser tools not available"

Install the required package:
```bash
pip install playwright
playwright install chromium
```

### "Domain not in whitelist"

Add the domain to `ALLOWED_DOMAINS` or request admin to add it.

### "Element not found"

1. Take a screenshot first to see the page layout
2. Make sure the CSS selector is correct
3. Element might not be visible - try JavaScript

### "Screenshot not saved"

Check that directories are writable:
```bash
chmod 755 vectorstore/ uploads/
```

## Performance Tips

1. **Use headless mode** (default) for speed: `headless=true`
2. **Take screenshots sparingly** (max 10 per session)
3. **Reuse browser sessions** when possible
4. **Set reasonable timeouts** for slow networks
5. **Use JavaScript** for bulk data extraction

## Integration with LLM Providers

### OpenAI

The browser tools work with any LLM that supports function calling, including:
- OpenAI GPT-4/GPT-4o
- Anthropic Claude
- Any OpenAI-compatible API

The system automatically provides the right tool definitions based on your provider.

### Adding Custom LLM Support

```python
from browser_integration import browser_integration

# Get tools for any provider
tools = browser_integration.get_tools_for_provider("my-provider")

# Pass to your LLM
response = my_llm.chat(
    messages=messages,
    tools=tools,
)
```

## Files Structure

```
browser_tools.py              # Core browser automation
browser_integration.py        # LLM integration layer
BROWSER_TOOLS.md             # Detailed documentation
BROWSER_TOOLS_README.md      # This file
test_browser_tools.py        # Tests and examples
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest test_browser_tools.py -v

# Run specific test
pytest test_browser_tools.py::TestBrowserToolsIntegration::test_open_browser -v

# Run with logging
pytest test_browser_tools.py -v -s
```

## Limitations

⚠️ **Known Limitations**

- **Performance**: Browser automation is slower than API calls
- **Dynamic Content**: Some JavaScript-heavy sites need time to load
- **CAPTCHAs**: Sites with CAPTCHA will block automated access
- **Cookies**: Browser sessions are isolated (no cookie persistence)
- **Screenshots**: Headless rendering may differ from desktop

## Best Practices

1. ✅ Always open browser before using other tools
2. ✅ Take screenshots to verify actions worked
3. ✅ Handle errors gracefully with retries
4. ✅ Close browser when done to free memory
5. ✅ Use clear, descriptive CSS selectors
6. ✅ Set reasonable timeouts for slow networks
7. ✅ Test with allowed domains first

## Future Enhancements

Planned features for future releases:
- [ ] PDF export from pages
- [ ] Web scraping with structured data
- [ ] Video recording of sessions
- [ ] Mobile device emulation
- [ ] Proxy support
- [ ] Cookie management
- [ ] Browser pool for concurrent sessions

## License

Browser tools are part of Salai and follow the same license.

Dependencies:
- **Playwright**: BSD License (Microsoft)
- **Selenium**: Apache 2.0 License

## Support

For issues or feature requests:

1. Check [BROWSER_TOOLS.md](BROWSER_TOOLS.md) for detailed docs
2. Review troubleshooting section above
3. Check test files for usage examples
4. Enable debug logging: `LOG_LEVEL=DEBUG`

## Security Notice

Browser automation tools are powerful. Use responsibly:

- Don't visit untrusted sites automatically
- Don't automate sites that explicitly forbid it
- Respect robots.txt and terms of service
- Don't use for spamming or abuse
- Monitor logs for security issues

## Quick Links

- [Detailed Documentation](BROWSER_TOOLS.md)
- [Integration Guide](browser_integration.py)
- [Test Examples](test_browser_tools.py)
- [Core Implementation](browser_tools.py)

---

**Ready to give your chatbot web browsing powers?** Install Playwright and start using browser tools in your chats! 🌐
