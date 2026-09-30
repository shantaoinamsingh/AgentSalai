# Browser Tools in Chat - Current Status & Implementation

## 📊 Current Status

### ✅ What Works
- Browser tools are installed and functional
- Agent executor is initialized
- Tools can be executed directly via Python
- Example script works perfectly
- System prompt mentions browser capabilities

### ⏳ What's Partially Done
- App detects browser tools availability
- App checks if provider supports tools
- Infrastructure for tool-use loop is in place
- Async event loop support added

### ❌ What's Missing for Full Chat Integration
The LLM in chat doesn't actually **use** the browser tools because:

1. **No Tool Parameters in Chat Call** - `providers.chat()` doesn't accept a `tools` parameter
2. **No Tool-Use Loop** - Chat doesn't handle when LLM says "I'll use a tool"
3. **No Tool Result Feedback** - Results from tools aren't sent back to LLM

---

## 🔧 Why Browser Tools Don't Work in Chat Yet

### Current Flow (What Happens Now)
```
User: "Search Google for Python tutorials"
  ↓
App: "Here's a search result..." (from knowledge base or general knowledge)
  ↓
No browser opened, no actual search performed
```

### Desired Flow (What Should Happen)
```
User: "Search Google for Python tutorials"
  ↓
LLM: "I'll search Google for you. Let me open a browser..."
  ↓
Agent: [opens browser → navigates → takes screenshot]
  ↓
LLM receives results: "I found X tutorials, here they are..."
  ↓
User sees actual search results
```

---

## 🛠️ What Needs to Be Done

### Step 1: Extend `providers.chat()` to Accept Tools

**File:** `providers.py`

**Current signature:**
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
) -> str:
```

**Needed signature:**
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
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
) -> Dict[str, Any]:  # Return dict with content AND tool_calls
```

**What to do:**
- For OpenAI: pass `tools` and `tool_choice` to the API call
- For Anthropic: pass `tools` to the API call
- Return `{"content": str, "tool_calls": List}` instead of just `str`

### Step 2: Implement Tool-Use Loop in `get_answer()`

**File:** `app.py` - modify `_get_answer_with_tools_async()`

```python
async def _get_answer_with_tools_async(...) -> str:
    """Execute LLM with tool support."""
    
    # Initial request with tools
    response = providers.chat(
        provider_id, model, messages,
        tools=tools,  # Pass tools!
        api_key=api_key,
        ...
    )
    
    # Check if LLM wants to use tools
    while response.get("tool_calls"):
        tool_calls = response["tool_calls"]
        
        # Execute tools
        tool_results = await agent_executor.execute_tool_calls(
            tool_calls,
            session_id=chat_id
        )
        
        # Add results to conversation
        messages.append({"role": "assistant", "content": response["content"]})
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": result["result"]
            })
        
        # Get next response
        response = providers.chat(
            provider_id, model, messages,
            tools=tools,  # Keep tools available
            api_key=api_key,
            ...
        )
    
    # Extract final text response
    return response.get("content", "No response")
```

---

## 📝 Implementation Checklist

### Phase 1: Update `providers.py` (30-45 min)
- [ ] Modify `chat()` signature to accept `tools` parameter
- [ ] Update OpenAI implementation to pass `tools`
- [ ] Update Anthropic implementation to pass `tools`
- [ ] Change return type to `Dict[str, Any]` with `content` and `tool_calls`
- [ ] Handle non-tool-using providers (return old format for compatibility)

### Phase 2: Update `app.py` (15-20 min)
- [ ] Implement tool-use loop in `_get_answer_with_tools_async()`
- [ ] Test with both OpenAI and Anthropic
- [ ] Verify tool results are properly formatted

### Phase 3: Test & Validate (15-20 min)
- [ ] Test in chat UI with browser tools
- [ ] Verify screenshot capture
- [ ] Test form filling
- [ ] Check error handling

---

## 💡 Quick Implementation (For OpenAI)

### In `providers.py`, for OpenAI (around line 194):

```python
def _openai_chat(
    spec: ProviderSpec,
    model: str,
    messages: List[Dict[str, str]],
    api_key: Optional[str],
    base_url: str,
    temperature: float,
    max_tokens: int,
    tools: Optional[List[Dict]] = None,
    tool_choice: str = "auto",
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    # Add tools if provided
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    
    try:
        r = requests.post(
            url, json=payload, 
            headers=_openai_headers(spec, api_key), 
            timeout=DEFAULT_TIMEOUT
        )
    except requests.exceptions.RequestException as e:
        raise ProviderError(f"Could not reach {spec.label}: {e}")

    if r.status_code != 200:
        raise ProviderError(_http_message(spec, r))

    try:
        response = r.json()
        content = response["choices"][0]["message"]["content"]
        tool_calls = response["choices"][0]["message"].get("tool_calls", [])
        
        # Return dict with content and tool_calls
        return {
            "content": content or "",
            "tool_calls": tool_calls
        }
    except (KeyError, IndexError, ValueError):
        raise ProviderError(f"{spec.label} returned an unexpected response shape.")
```

---

## 🚀 How to Proceed

### Option 1: Implement Now (Recommended if you have 1-2 hours)
Follow the checklist above and enable full browser tool support in chat.

### Option 2: Use API Endpoint (Quick, works now)
Create a `/api/browser/execute` endpoint that takes actions:
```
POST /api/browser/execute
{ "action": "navigate", "params": {"url": "google.com", "session_id": "s1"} }
```

### Option 3: Direct Tool Usage (Works now)
Keep using the Python examples and direct tool execution.

---

## 📝 Current App State

| Component | Status | Notes |
|-----------|--------|-------|
| Browser tools | ✅ Installed | Playwright ready |
| Agent executor | ✅ Ready | Can execute tools |
| Chat integration | ⏳ Partial | Detects tools, not using them |
| Tool-use loop | ❌ Missing | Needs implementation |
| LLM provider mods | ❌ Missing | Needs tool parameter support |

---

## 💬 Example Chat (Once Fully Implemented)

```
User: Search for "Python programming" on Google