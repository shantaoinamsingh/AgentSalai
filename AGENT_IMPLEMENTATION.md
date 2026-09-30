# Agent Implementation Guide

Complete guide to implementing browser tools agent into Salai for LLM tool use.

## Overview

The agent system allows LLM providers that support function calling (OpenAI, Anthropic) to:
- Decide when to use browser tools
- Execute tools during conversation
- Process tool results and provide answers

## Architecture

```
User Message
    ↓
LLM Provider (OpenAI/Anthropic/etc)
    ↓
[Does LLM decide to use a tool?]
    ├─ YES → Agent Executor
    │         ├─ Parse tool call
    │         ├─ Execute tool (browser_tools)
    │         └─ Return result to LLM
    │
    └─ NO → Return answer to user
```

## Files Structure

```
agent.py                    # Core agent executor
browser_integration.py      # Browser tools integration
browser_tools.py            # Browser automation
providers.py                # Updated with tool support
```

## Setup & Installation

### Step 1: Install Dependencies

```bash
# Core (already have)
pip install Flask Flask-SocketIO

# Browser tools (optional but recommended)
pip install playwright
playwright install chromium

# Or use Selenium
pip install selenium
```

### Step 2: Enable Tools in Your Code

The agent is automatically initialized. Tools are available if:
- Browser library is installed (Playwright or Selenium)
- `agent.py` is in the project
- `from agent import agent_executor` imports successfully

### Step 3: Test Browser Tools

```bash
# Run tests
pytest test_browser_tools.py -v

# Or test in Python
python -c "from agent import agent_executor; print(agent_executor.get_available_tools())"
```

## Usage Examples

### Example 1: OpenAI with Tool Use

```python
from openai import OpenAI
from agent import agent_executor
import asyncio
import json

client = OpenAI(api_key="your-key")

async def chat_with_tools():
    # Get available tools
    tools = agent_executor.get_available_tools()
    
    messages = [
        {"role": "user", "content": "Search Google for Python tutorials"}
    ]
    
    # Create chat completion with tools
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # Let model decide
    )
    
    # Handle tool calls
    while response.choices[0].finish_reason == "tool_calls":
        tool_calls = response.choices[0].message.tool_calls
        
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
            ]
        )
        
        # Add assistant message with tool calls
        messages.append(response.choices[0].message)
        
        # Add tool results
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": result["result"]
            })
        
        # Continue conversation
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            tools=tools,
        )
    
    # Get final answer
    return response.choices[0].message.content

# Run it
answer = asyncio.run(chat_with_tools())
print(answer)
```

### Example 2: Anthropic (Claude) with Tool Use

```python
from anthropic import Anthropic
from agent import agent_executor
import asyncio
import json

client = Anthropic(api_key="your-key")

async def chat_with_tools():
    tools = agent_executor.get_available_tools()
    
    messages = [
        {"role": "user", "content": "Search GitHub for Salai projects"}
    ]
    
    # Create message with tools
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        tools=tools,
        messages=messages,
    )
    
    # Process response
    while response.stop_reason == "tool_use":
        # Extract tool uses
        tool_uses = [block for block in response.content if block.type == "tool_use"]
        
        # Execute tools
        tool_results = await agent_executor.execute_tool_calls(
            [
                {
                    "id": use.id,
                    "name": use.name,
                    "arguments": use.input,
                }
                for use in tool_uses
            ]
        )
        
        # Build next message
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
        
        # Continue conversation
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            tools=tools,
            messages=messages,
        )
    
    # Get final text response
    text_blocks = [block for block in response.content if block.type == "text"]
    return "".join([block.text for block in text_blocks])

# Run it
answer = asyncio.run(chat_with_tools())
print(answer)
```

### Example 3: Salai Integration (Flask + SocketIO)

Here's how to integrate into the existing Salai app:

```python
# In app.py - modify get_answer function

import asyncio
from agent import agent_executor
import json

async def get_answer_with_tools(user_input: str, chat_id: str, user_id: str) -> str:
    """Run one conversation turn with tool support."""
    
    user_input = (user_input or "").strip()
    if not user_input:
        return "Please enter a question."

    messages, meta = build_messages(chat_id, user_input, user_id)
    settings = settings_store.get(user_id)
    provider_id = settings["provider"]

    # Add tools if provider supports them
    tools = None
    if providers.supports_tools(provider_id):
        tools = agent_executor.get_available_tools()

    # Initial request to LLM
    response = providers.chat_with_tools(
        provider_id,
        settings["model"],
        messages,
        tools=tools,
        api_key=settings_store.api_key(user_id, provider_id),
        base_url=settings["base_url"],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
    )

    # Handle tool calls if any
    while response.get("tool_calls"):
        tool_calls = response["tool_calls"]
        
        # Execute tools
        tool_results = await agent_executor.execute_tool_calls(
            tool_calls,
            session_id=chat_id,
        )
        
        # Add tool results to conversation
        messages.append({"role": "assistant", "content": response.get("content", "")})
        
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": result["result"]
            })
        
        # Get next response from LLM
        response = providers.chat_with_tools(
            provider_id,
            settings["model"],
            messages,
            tools=tools,
            api_key=settings_store.api_key(user_id, provider_id),
            base_url=settings["base_url"],
            temperature=settings["temperature"],
            max_tokens=settings["max_tokens"],
        )

    # Get final answer
    answer = response.get("content", "Unable to generate response")

    # Store conversation
    chat_store.append_message(chat_id, "user", user_input)
    chat_store.append_message(chat_id, "assistant", answer)
    
    return answer
```

## Supported Providers

| Provider | Tool Support | Status |
|----------|--------------|--------|
| OpenAI (GPT-4, o1) | ✅ Yes | Ready |
| Anthropic (Claude) | ✅ Yes | Ready |
| OpenRouter | ✅ Yes (if using OpenAI/Anthropic) | Ready |
| Self-hosted | ⚠️ Depends | Check API |

## Tool Definitions

Available tools are automatically detected from:

1. **Browser Tools** (if Playwright/Selenium installed)
   - `open_browser`
   - `navigate`
   - `click_element`
   - `fill_input`
   - `take_screenshot`
   - `execute_js`
   - `close_browser`

Access tools:

```python
from agent import agent_executor

# Get all tools
tools = agent_executor.get_available_tools()

# Check if tools are available
if tools:
    print(f"Available {len(tools)} tools")
```

## Conversation Flow

### Without Tool Use
```
User: "What is Python?"
  ↓
LLM decides: No tools needed
  ↓
LLM: "Python is a programming language..."
```

### With Tool Use
```
User: "Search Google for Python tutorials"
  ↓
LLM decides: I need to use browser tools
  ↓
1. Call open_browser()
2. Call navigate("google.com")
3. Call fill_input(selector, "Python tutorials")
4. Call click_element(selector)
5. Call take_screenshot()
  ↓
LLM receives results, processes them
  ↓
LLM: "I found several Python tutorials..."
```

## Error Handling

The agent handles errors gracefully:

```python
# Tool execution errors are caught
results = await agent_executor.execute_tool_calls(tool_calls)

# Each result has status
for result in results:
    if result["status"] == "error":
        print(f"Tool {result['tool_name']} failed: {result['result']}")
    else:
        print(f"Tool {result['tool_name']} succeeded")
```

## Security Considerations

1. **Tool Restrictions**: Only approved tools are available
2. **Domain Whitelist**: Browser can only visit whitelisted domains
3. **Tool Validation**: All tool inputs are validated
4. **Logging**: All tool use is logged for auditing
5. **Resource Limits**: Prevents abuse (10 screenshots/session, 1MB limit)

## Performance Tips

1. **Reuse Sessions**: Keep browser open for multiple operations
2. **Minimal Screenshots**: Take screenshots only when needed
3. **JavaScript for Data**: Use JS to extract data efficiently
4. **Headless Mode**: Always use headless for speed
5. **Timeout Tuning**: Adjust timeouts based on network

## Testing Tool Use

```bash
# Test agent executor
pytest test_browser_tools.py -v

# Test in Python shell
python
>>> from agent import agent_executor
>>> tools = agent_executor.get_available_tools()
>>> len(tools)  # Should show number of available tools
```

## Debugging

Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Now run your code
# You'll see detailed logs of tool execution
```

## Limitations

1. **LLM Decision**: LLM decides when to use tools, not the user
2. **Latency**: Browser operations add 5-30 seconds per action
3. **Single Browser**: One browser per session (parallel needs work)
4. **Dynamic Content**: Some JS-heavy sites may need wait time
5. **CAPTCHAs**: Automated access will be blocked by CAPTCHA

## Troubleshooting

### "Tools not available"

```bash
# Install browser library
pip install playwright
playwright install chromium

# Verify in Python
from agent import agent_executor
print(len(agent_executor.get_available_tools()))
```

### "Tool call fails silently"

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run your code again
```

### "LLM doesn't use tools"

- Some models may not use tools - try explicit prompt
- Check if provider supports tools: `providers.supports_tools(provider_id)`
- Verify tools are passed to LLM in request

## Next Steps

1. ✅ Install Playwright or Selenium
2. ✅ Test agent with browser tools
3. ✅ Integrate into your LLM requests
4. ✅ Monitor tool usage and logs
5. ✅ Adjust configurations as needed

## Support

For questions or issues:
1. Check this documentation
2. Review example code above
3. Check test files for patterns
4. Enable debug logging for troubleshooting
5. Review browser tool documentation
