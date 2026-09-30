# Browser Automation Tools for Salai

Enable your Salai chatbot to browse the web, interact with websites, and perform actions based on chat commands.

## Overview

The browser automation tools allow the LLM to:
- 🌐 Open and navigate web pages
- 🖱️ Click buttons, links, and interactive elements
- ⌨️ Fill forms and text inputs
- 📸 Take screenshots to verify actions
- 🔧 Execute JavaScript on web pages
- 📊 Extract data from websites

## Installation

### Option 1: Playwright (Recommended)

```bash
pip install playwright
playwright install chromium
```

### Option 2: Selenium

```bash
pip install selenium
# Download ChromeDriver from https://chromedriver.chromium.org/
```

### Or Both

```bash
pip install playwright selenium
```

The system automatically detects which library is available and uses Playwright first, falling back to Selenium.

## Quick Start

### 1. Enable Browser Tools in Chat

Users can request the browser to be opened:

```
User: "Open a web browser and search Google for Python tutorials"
Assistant: I'll help you search for Python tutorials. Let me open a browser...
[Opens browser → navigates to google.com → fills search → clicks search]
Here are the top Python tutorials I found...
```

### 2. In Code

```python
from browser_integration import browser_integration
import asyncio

async def example():
    # Get available tools
    tools = browser_integration.get_tools_for_provider("openai")
    
    # Execute browser actions
    result = await browser_integration.execute_tool(
        "open_browser",
        {"session_id": "user123"}
    )
    print(result)  # "Browser opened successfully"
    
    # Navigate to a website
    result = await browser_integration.execute_tool(
        "navigate",
        {"session_id": "user123", "url": "python.org"}
    )
    
    # Take screenshot
    result = await browser_integration.execute_tool(
        "take_screenshot",
        {"session_id": "user123", "name": "python_site"}
    )

asyncio.run(example())
```

## Available Tools

### 1. `open_browser`

Open a new browser instance.

**Parameters:**
- `session_id` (string, required): Unique session identifier
- `headless` (boolean, optional): Run without GUI (default: true)

**Example:**
```
open_browser(session_id="user_123", headless=true)
```

**Response:**
```
✓ Browser opened successfully (Playwright, headless=true)
```

### 2. `navigate`

Navigate to a URL.

**Parameters:**
- `session_id` (string, required): Browser session ID
- `url` (string, required): URL to visit (must be whitelisted)

**Example:**
```
navigate(session_id="user_123", url="google.com")
```

**Response:**
```
✓ Navigated to https://google.com

Page content preview:
Google Search
Images Maps News More Settings Tools...
```

### 3. `click_element`

Click a button or link using CSS selector.

**Parameters:**
- `session_id` (string, required): Browser session ID
- `selector` (string, required): CSS selector

**Example:**
```
click_element(session_id="user_123", selector="#search-button")
```

**Selector Examples:**
- By ID: `#button-id`
- By class: `.btn-primary`
- By tag: `button`
- By attribute: `[type="submit"]`
- Complex: `div.search-form button[type="submit"]`

### 4. `fill_input`

Fill a text input field.

**Parameters:**
- `session_id` (string, required): Browser session ID
- `selector` (string, required): CSS selector for input
- `text` (string, required): Text to enter

**Example:**
```
fill_input(session_id="user_123", selector="input[name='q']", text="Python tutorial")
```

### 5. `take_screenshot`

Capture a screenshot of the current page.

**Parameters:**
- `session_id` (string, required): Browser session ID
- `name` (string, optional): Screenshot name

**Example:**
```
take_screenshot(session_id="user_123", name="search_results")
```

**Response:**
```
✓ Screenshot saved: search_results (524288 bytes)
```

### 6. `execute_js`

Execute JavaScript on the page.

**Parameters:**
- `session_id` (string, required): Browser session ID
- `script` (string, required): JavaScript code

**Example:**
```
execute_js(session_id="user_123", script="document.title")
```

**Response:**
```
✓ JavaScript executed: Google Search
```

**JavaScript Examples:**
```javascript
// Get page title
document.title

// Get all links
Array.from(document.querySelectorAll('a')).map(a => a.href)

// Get page height
document.body.scrollHeight

// Get text content
document.body.innerText
```

### 7. `close_browser`

Close a browser instance and free resources.

**Parameters:**
- `session_id` (string, required): Browser session ID

**Example:**
```
close_browser(session_id="user_123")
```

**Response:**
```
✓ Browser closed. Captured 3 screenshots.
```

## Use Cases

### 1. Web Search & Information Gathering

```
User: "Find the latest news about Python 3.12"
Assistant: I'll search for that information for you.
1. Open browser → Navigate to Google
2. Fill search box with "Python 3.12 release news"
3. Click search button
4. Extract results from page

Here are the latest news articles about Python 3.12:
- Release notes showing new features
- Performance improvements...
```

### 2. Data Extraction

```
User: "Get all links from the Python documentation page"
Assistant: I'll extract all links from the Python docs.
1. Navigate to python.org/doc
2. Take screenshot to see the page
3. Execute JavaScript to get all links

Found 42 links:
- Getting Started - https://...
- Tutorial - https://...
- Library Reference - https://...
```

### 3. Form Filling & Submission

```
User: "Sign up for the newsletter with email john@example.com"
Assistant: I'll fill out the newsletter signup form.
1. Navigate to website
2. Find email input field
3. Fill with "john@example.com"
4. Click submit button
5. Take screenshot to confirm

✓ Successfully signed up for the newsletter!
```

### 4. E-commerce Tasks

```
User: "Add a Python book to the shopping cart"
Assistant: I'll help you add a book.
1. Search for Python books
2. Click on the first book result
3. Click "Add to Cart" button
4. Take screenshot of cart

✓ Added "Python Programming" to cart. Current total: $29.99
```

### 5. Website Monitoring

```
User: "Check if the website is up and take a screenshot"
Assistant: I'll check the website status.
1. Navigate to the URL
2. Take screenshot
3. Get page title and content

✓ Website is up and running. Screenshot saved.
```

## Security & Limitations

### ✅ Security Features

1. **Domain Whitelist**: Only allowed domains can be visited
   - Default whitelisted: google.com, wikipedia.org, github.com, etc.
   - Configure: Edit `ALLOWED_DOMAINS` in `browser_tools.py`

2. **URL Validation**:
   - Blocks malicious keywords
   - Requires valid scheme (http:// or https://)
   - Prevents known attack vectors

3. **Resource Limits**:
   - Max 10 screenshots per session
   - 1MB page content limit
   - 30-second page load timeout
   - JavaScript safety checks

4. **Activity Logging**:
   - All browser activities are logged
   - Security events tracked
   - Useful for debugging and auditing

5. **Restricted JavaScript**:
   - Blocks access to localStorage, sessionStorage, cookies
   - Prevents credential theft
   - Only safe operations allowed

### ⚠️ Limitations

1. **Performance**: Browser automation can be slow (network latency)
2. **Headless Mode**: Screenshots may look different from desktop
3. **JavaScript Heavy Sites**: Some modern apps may need time to load
4. **CAPTCHAs**: Will block automation on protected sites
5. **Session Management**: Browser cookies/sessions are isolated

## Configuration

### Add Custom Whitelisted Domains

Edit `browser_tools.py`:

```python
ALLOWED_DOMAINS = [
    "example.com",
    "google.com",
    "wikipedia.org",
    "github.com",
    "your-domain.com",  # Add your domain here
]
```

### Adjust Resource Limits

```python
MAX_RETRIES = 3                  # Retry failed navigations
RETRY_DELAY = 1                  # Seconds between retries
PAGE_LOAD_TIMEOUT = 30           # Seconds to wait for page load
MAX_SCREENSHOTS = 10             # Screenshots per session
MAX_PAGE_SIZE = 1_000_000        # 1MB content limit
```

### Change Browser Engine

The system auto-detects and uses Playwright if available, else Selenium:

```python
# Force Playwright
import playwright.async_api

# Force Selenium
import selenium.webdriver
```

## Troubleshooting

### "Browser tools not available"

Install required package:
```bash
pip install playwright
playwright install chromium
```

Or:
```bash
pip install selenium
```

### "Domain not in whitelist"

Add the domain to `ALLOWED_DOMAINS` in `browser_tools.py`, or use an already-whitelisted domain.

### "Failed to navigate to URL"

1. Check internet connection
2. Verify URL is correct
3. Site might block automated access
4. Try a different URL

### "Element not found"

1. Take a screenshot first to see the page
2. Verify CSS selector is correct
3. Element might not be visible (use JavaScript)
4. Page might not be fully loaded

### "Screenshots not saving"

Check file system permissions:
```bash
chmod 755 ./vectorstore/
chmod 755 ./uploads/
```

## Integration with LLM Providers

### OpenAI

```python
from openai import OpenAI
from browser_integration import browser_integration

client = OpenAI()

# Add browser tools to function calls
tools = browser_integration.get_tools_for_provider("openai")

response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": "Search for Python tutorials"}],
    tools=tools,
    tool_choice="auto",
)

# Handle tool calls from the model
for tool_call in response.tool_calls:
    result = await browser_integration.execute_tool(
        tool_call.function.name,
        json.loads(tool_call.function.arguments),
    )
```

### Anthropic (Claude)

```python
from anthropic import Anthropic
from browser_integration import browser_integration

client = Anthropic()

# Add browser tools to tool definitions
tools = browser_integration.get_tools_for_provider("anthropic")

response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=4096,
    tools=tools,
    messages=[{"role": "user", "content": "Search for Python tutorials"}],
)

# Handle tool use blocks from Claude
```

## Best Practices

1. **Always Open First**: Open browser before using other tools
2. **Take Screenshots**: Verify actions with screenshots
3. **Handle Errors**: Retry failed actions gracefully
4. **Close Browser**: Clean up resources when done
5. **Clear Selectors**: Use simple, unique CSS selectors
6. **Describe Actions**: Explain what you're doing to the user
7. **Set User Expectations**: Browser automation is slower than API calls

## Performance Tips

1. Use `headless=true` (default) for faster execution
2. Reuse browser sessions when possible
3. Minimize screenshots (take only when needed)
4. Use JavaScript for bulk data extraction
5. Set reasonable timeouts

## License & Attribution

Browser automation tools integrate:
- **Playwright**: By Microsoft (BSD)
- **Selenium**: By Selenium Project (Apache 2.0)

## Support & Contributing

For issues or feature requests:
1. Check the troubleshooting section above
2. Review browser tool logs
3. Open an issue with detailed error messages
4. Include screenshots and steps to reproduce

## Security Considerations

### For Administrators

1. Regularly audit whitelisted domains
2. Monitor browser activity logs
3. Update browser automation libraries
4. Set resource limits appropriately
5. Implement rate limiting if needed

### For Users

1. Don't share browser sessions with untrusted users
2. Be aware browser automation can be slow
3. Don't expect perfect JavaScript rendering
4. Some sites explicitly block automation
5. Screenshot content is preserved in memory

## Future Enhancements

Potential features for future versions:
- [ ] PDF export from web pages
- [ ] Web scraping with structured data extraction
- [ ] Video recording of browser sessions
- [ ] Mobile device emulation
- [ ] Proxy support for anonymity
- [ ] Custom headers support
- [ ] Cookie management
- [ ] Browser pool management