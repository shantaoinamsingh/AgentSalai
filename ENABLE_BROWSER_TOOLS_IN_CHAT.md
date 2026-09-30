# Enable Browser Tools in Chat

Browser tools are now partially integrated into the Salai chatbot. Here's the status and how to fully enable them.

## Current Status

✅ **What's Ready:**
- Browser tools installed and working
- Agent executor integrated into app.py
- System prompt updated to encourage tool use
- Tools are loaded and available

⏳ **What Needs Manual Implementation:**
- Full tool-use loop in the chat flow
- Handling tool calls from LLM responses
- Processing tool results and continuing conversation

## Why Partial Integration?

The current Flask/SocketIO chat system is designed for single-turn LLM calls without tool use loops. Fully implementing tool use requires:

1. Detecting when the LLM wants to use a tool
2. Executing the tool asynchronously
3. Feeding results back to the LLM
4. Continuing the conversation

This is a multi-turn process that needs to be integrated into the chat loop.

## How to Use Browser Tools Right Now

### Option 1: Direct Python (Recommended for Now)

Use the browser tools directly in Python:

```python
import asyncio
from agent import agent_executor

async def search_python():
    # Open browser
    await agent_executor.execute_tool(
        "open_browser",
        {"session_id": "search_1"}
    )
    
    # Navigate to Google
    await agent_executor.execute_tool(
        "navigate",
        {"session_id": "search_1", "url": "google.com"}
    )
    
    # Fill search box
    await agent_executor.execute_tool(
        "fill_input",
        {"session_id": "search_1", "selector": "input[name='q']", "text": "Python tutorials"}
    )
    
    # Click search
    await agent_executor.execute_tool(
        "click_element",
        {"session_id": "search_1", "selector": "input[value='Google Search']"}
    )
    
    # Take screenshot
    await agent_executor.execute_tool(
        "take_screenshot",
        {"session_id": "search_1", "name": "search_results"}
    )
    
    # Close
    await agent_executor.execute_tool(
        "close_browser",
        {"session_id": "search_1"}
    )

asyncio.run(search_python())
```

### Option 2: Use Example Script

Run the example:
```bash
python example_agent_usage.py
```

### Option 3: Full Chat Integration (Advanced)

Extend `get_answer()` to handle tool calls. Here's a skeleton:

```python
async def get_answer_with_tools(user_input: str, chat_id: str, user_id: str) -> str:
    """Handle tool use in conversations."""
    
    messages, meta = build_messages(chat_id, user_input, user_id)
    settings = settings_store.get(user_id)
    provider_id = settings["provider"]
    
    # Get tools if provider supports them
    tools = None
    if providers.supports_tools(provider_id):
        tools = agent_executor.get_available_tools()
    
    # Initial request (requires extending providers.chat to accept tools parameter)
    response = providers.chat(
        provider_id,
        settings["model"],
        messages,
        tools=tools,  # Would need to add this to providers.chat()
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
        
        # Add results to conversation
        messages.append({"role": "assistant", "content": response.get("content", "")})
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": result["result"]
            })
        
        # Continue conversation
        response = providers.chat(
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

## Steps to Full Integration

1. **Extend providers.chat()** to accept and handle tools parameter
2. **Update get_answer()** to use async execution with tool loops
3. **Modify SocketIO handler** to await async get_answer
4. **Test** with different LLM providers (OpenAI, Anthropic)

## What's Needed in Code

### 1. In `providers.py`, extend `chat()`:

```python
def chat(
    provider_id: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    tools: Optional[List[Dict]] = None,  # ADD THIS
    tool_choice: str = "auto",            # ADD THIS
) -> Dict[str, Any]:  # CHANGE RETURN TYPE
    """Send completion request with optional tool use."""
    # Implementation would return dict with content and tool_calls
```

### 2. In `app.py`, make get_answer async:

```python
async def get_answer(user_input: str, chat_id: str, user_id: str) -> str:
    # ... existing code ...
    
    # Use tool loop
    while response.get("tool_calls"):
        # ... handle tools ...
    
    return final_answer
```

## Quick Path Forward

If you want browser tools to work in the chat immediately:

1. ✅ Browser tools are ready
2. ✅ Agent executor is ready
3. Create a simple test endpoint like `/api/browser/search` that:
   - Takes a search query
   - Uses agent_executor to search
   - Returns results

Example:
```python
@app.route("/api/browser/search", methods=["POST"])
def browser_search():
    query = request.json.get("query", "")
    results = asyncio.run(search_web(query))
    return jsonify({"results": results})

async def search_web(query):
    session_id = f"search_{uuid.uuid4().hex}"
    
    # Open browser, search, close
    await agent_executor.execute_tool("open_browser", {"session_id": session_id})
    # ... execute search ...
    
    return results
```

## Current Limitations

- Browser tools in Flask requires async handlers
- SocketIO events are currently synchronous
- Need to extend providers.py to support tool parameters

## What Works Right Now

✅ Browser automation via direct Python calls  
✅ Web navigation and screenshots  
✅ Form filling and JavaScript execution  
✅ All 7 browser tools functional  

## What Needs Work

❌ Automatic tool use in chat (requires async refactor)  
❌ LLM deciding when to use tools (requires provider.chat() extension)  
❌ Multi-turn tool conversations (requires loop implementation)  

## Next Steps

Choose your approach:

**Simple:** Use `/api/browser/search` endpoint for specific browser tasks
**Medium:** Extend one endpoint to handle tool use loops
**Full:** Refactor entire chat system to support async tool use

All code is ready in `agent.py`, `browser_tools.py`, and `example_agent_usage.py`.

---

**TL;DR:** Browser tools work perfectly in Python. The chat UI would need the tool-use loop added to `get_answer()` to fully integrate them. You can use them directly right now with `example_agent_usage.py` or create a dedicated endpoint.
