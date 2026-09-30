#!/usr/bin/env python3
"""Pluggable LLM providers.

Users can pick a provider and supply their own credentials. Every provider
takes and returns the same shapes:

    chat(messages, ...) -> str            messages are OpenAI-style dicts
    list_models(...)    -> list[str]
    generate_image(...) -> {"b64" | "url"}

Credentials are passed in per call. Nothing in this module reads or writes them
from disk; see settings_store.py for how they are held.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = int(os.getenv("LLM_TIMEOUT_SECONDS" , 120))


class ProviderError(RuntimeError):
    """A provider call failed in a way worth showing the user."""


@dataclass
class ProviderSpec:
    id: str
    label: str
    kind: str  # openai | anthropic
    needs_key: bool
    description: str
    models: List[str] = field(default_factory=list)
    default_model: str = ""
    base_url: str = ""
    editable_base_url: bool = False
    can_list_models: bool = False
    supports_images: bool = False
    key_url: str = ""

    def public(self) -> Dict[str, Any]:
        """Everything the settings UI needs. Never includes credentials."""
        return {
            "id": self.id,
            "label": self.label,
            "needs_key": self.needs_key,
            "description": self.description,
            "models": self.models,
            "default_model": self.default_model,
            "base_url": self.base_url,
            "editable_base_url": self.editable_base_url,
            "can_list_models": self.can_list_models,
            "supports_images": self.supports_images,
            "key_url": self.key_url,
        }


# Model lists are starting points for the dropdown, not a whitelist -- users can
# type any model id their provider serves.
PROVIDERS: Dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        id="openai",
        label="OpenAI",
        kind="openai",
        needs_key=True,
        description="Your own OpenAI account. Supports image generation.",
        models=["gpt-4o", "gpt-4o-mini", "gpt-4.1", "o3", "o4-mini"],
        default_model="gpt-4o",
        base_url="https://api.openai.com/v1",
        can_list_models=True,
        supports_images=True,
        key_url="https://platform.openai.com/api-keys",
    ),
    "anthropic": ProviderSpec(
        id="anthropic",
        label="Anthropic (Claude)",
        kind="anthropic",
        needs_key=True,
        description="Claude models direct from Anthropic.",
        models=[
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-haiku-4-5",
            "claude-fable-5-1",
        ],
        default_model="claude-opus-5",
        base_url="https://api.anthropic.com",
        can_list_models=True,
        key_url="https://console.anthropic.com/settings/keys",
    ),
    "gemini": ProviderSpec(
        id="gemini",
        label="Google Gemini",
        kind="openai",
        needs_key=True,
        description="Gemini models direct from Google AI Studio.",
        models=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"],
        default_model="gemini-2.5-flash",
        # Google's OpenAI-compatible surface: same chat/tools/models shapes.
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        can_list_models=True,
        key_url="https://aistudio.google.com/apikey",
    ),
    "openrouter": ProviderSpec(
        id="openrouter",
        label="OpenRouter",
        kind="openai",
        needs_key=True,
        description="One key for hundreds of models across providers, "
        "including Claude, Gemini, Llama and Mistral.",
        models=[
            "anthropic/claude-opus-4.1",
            "openai/gpt-4o",
            "google/gemini-2.5-pro",
            "meta-llama/llama-3.3-70b-instruct",
            "mistralai/mistral-large",
            "deepseek/deepseek-chat",
        ],
        default_model="anthropic/claude-opus-4.1",
        base_url="https://openrouter.ai/api/v1",
        can_list_models=True,
        key_url="https://openrouter.ai/keys",
    ),
    "compatible": ProviderSpec(
        id="compatible",
        label="Self-hosted / OpenAI-compatible",
        kind="openai",
        needs_key=False,
        description="Any server exposing an OpenAI-compatible /chat/completions "
        "endpoint: Ollama, vLLM, LM Studio, llama.cpp, Groq, Together.",
        models=["llama3.1", "qwen2.5", "mistral", "phi3"],
        default_model="llama3.1",
        base_url="http://localhost:11434/v1",
        editable_base_url=True,
        can_list_models=True,
    ),
}

DEFAULT_PROVIDER = "openai"


def get_spec(provider_id: str) -> ProviderSpec:
    spec = PROVIDERS.get(provider_id)
    if spec is None:
        raise ProviderError(f"Unknown provider: {provider_id!r}")
    return spec


def public_catalog() -> List[Dict[str, Any]]:
    return [p.public() for p in PROVIDERS.values()]


# ─── message shaping ──────────────────────────────────────────────────────────


def _split_system(messages: List[Dict[str, Any]]) -> tuple:
    """Separate system text from the turn list (Anthropic keeps them apart).

    Also converts OpenAI-style tool turns into Anthropic tool_use/tool_result blocks.
    """
    system_chunks = [m["content"] for m in messages if m.get("role") == "system"]
    turns: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m.get("content") or "",
            }
            # Anthropic wants all results for one assistant turn in a single user turn.
            if turns and turns[-1]["role"] == "user" and isinstance(turns[-1]["content"], list):
                turns[-1]["content"].append(block)
            else:
                turns.append({"role": "user", "content": [block]})
        elif role == "assistant" and m.get("_anthropic_content") is not None:
            turns.append({"role": "assistant", "content": m["_anthropic_content"]})
        elif role in ("user", "assistant"):
            turns.append({"role": role, "content": m["content"]})
    return "\n\n".join(system_chunks), turns


# ─── OpenAI-compatible (OpenAI, OpenRouter, Ollama, vLLM, ...) ────────────────


def _openai_headers(spec: ProviderSpec, api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if spec.id == "openrouter":
        # OpenRouter asks callers to identify themselves; harmless elsewhere.
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE", "http://localhost:5000")
        headers["X-Title"] = "Salai"
    return headers


def _openai_chat(
    spec: ProviderSpec,
    model: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str],
    base_url: str,
    temperature: float,
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    # Keys starting with "_" carry provider-private state (e.g. Anthropic blocks).
    clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
    payload: Dict[str, Any] = {
        "model": model,
        "messages": clean,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]
        payload["tool_choice"] = "auto"
    try:
        r = requests.post(
            url, json=payload, headers=_openai_headers(spec, api_key), timeout=DEFAULT_TIMEOUT
        )
    except requests.exceptions.Timeout:
        raise ProviderError(f"{spec.label} timed out after {DEFAULT_TIMEOUT}s.")
    except requests.exceptions.RequestException as e:
        raise ProviderError(f"Could not reach {spec.label}: {e}")

    if r.status_code == 404:
        raise ProviderError(_http_message(spec, r) + _suggest_models(spec, model, api_key, base_url))
    if r.status_code != 200:
        raise ProviderError(_http_message(spec, r))

    try:
        msg = r.json()["choices"][0]["message"]
    except (KeyError, IndexError, ValueError):
        raise ProviderError(f"{spec.label} returned an unexpected response shape.")

    raw_calls = msg.get("tool_calls") or []
    calls = []
    for c in raw_calls:
        fn = c.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {}
        calls.append({"id": c.get("id"), "name": fn.get("name"), "arguments": args})

    content = msg.get("content") or ""
    assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content or None}
    if raw_calls:
        assistant_msg["tool_calls"] = raw_calls
    return {"content": content, "tool_calls": calls, "message": assistant_msg}


def _suggest_models(spec: ProviderSpec, model: str, api_key: Optional[str], base_url: str) -> str:
    """On a model 404, name a few models this key can actually use."""
    if not spec.can_list_models:
        return ""
    try:
        ids = _openai_models(spec, api_key, base_url)
    except Exception:
        return ""
    family = model.split("-")[0].lower()
    skip = ("embedding", "tts", "image", "audio", "aqa", "live")
    matches = [i for i in ids if i.lower().startswith(family) and not any(s in i.lower() for s in skip)]
    picks = (matches or ids)[:8]
    return f" Models available to this key include: {', '.join(picks)}. Pick one in Settings." if picks else ""


def _openai_models(spec: ProviderSpec, api_key: Optional[str], base_url: str) -> List[str]:
    url = f"{base_url.rstrip('/')}/models"
    r = requests.get(url, headers=_openai_headers(spec, api_key), timeout=30)
    if r.status_code != 200:
        raise ProviderError(_http_message(spec, r))
    data = r.json().get("data", [])
    # Gemini lists ids as "models/gemini-..."; chat expects the bare name.
    ids = sorted(
        m["id"].removeprefix("models/") for m in data if isinstance(m, dict) and m.get("id")
    )
    return ids


# ─── Anthropic ────────────────────────────────────────────────────────────────


def _anthropic_client(api_key: str):
    try:
        import anthropic
    except ImportError:
        raise ProviderError(
            "The 'anthropic' package is not installed. Run: pip install anthropic"
        )
    return anthropic, anthropic.Anthropic(api_key=api_key, timeout=float(DEFAULT_TIMEOUT))


def _anthropic_chat(
    model: str,
    messages: List[Dict[str, str]],
    api_key: str,
    temperature: float,
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    anthropic, client = _anthropic_client(api_key)
    system, turns = _split_system(messages)
    if not turns:
        raise ProviderError("Anthropic requires at least one user message.")

    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": turns,
    }
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
            }
            for t in tools
        ]
    # Sampling params are rejected on the newest models (they always think), so
    # only send temperature to models that still accept it.
    if not _is_adaptive_thinking_model(model):
        kwargs["temperature"] = temperature

    try:
        response = client.messages.create(**kwargs)
    except anthropic.AuthenticationError:
        raise ProviderError("Anthropic rejected that API key.")
    except anthropic.PermissionDeniedError:
        raise ProviderError("That Anthropic key lacks permission for this model.")
    except anthropic.NotFoundError:
        raise ProviderError(f"Anthropic has no model called {model!r}.")
    except anthropic.RateLimitError:
        raise ProviderError("Anthropic rate limit reached. Try again shortly.")
    except anthropic.APIStatusError as e:
        raise ProviderError(f"Anthropic error {e.status_code}: {getattr(e, 'message', '')}")
    except anthropic.APIConnectionError:
        raise ProviderError("Could not reach Anthropic. Check the network.")

    if getattr(response, "stop_reason", None) == "refusal":
        raise ProviderError("Claude declined to answer that request.")

    # Responses can carry thinking blocks alongside text; keep only the text.
    parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    calls = [
        {"id": b.id, "name": b.name, "arguments": dict(b.input or {})}
        for b in response.content
        if getattr(b, "type", None) == "tool_use"
    ]
    if not parts and not calls:
        raise ProviderError("Claude returned no text content.")
    content = "".join(parts)
    # The full block list (thinking included) must be echoed back on tool turns.
    assistant_msg = {"role": "assistant", "content": content, "_anthropic_content": response.content}
    return {"content": content, "tool_calls": calls, "message": assistant_msg}


def _is_adaptive_thinking_model(model: str) -> bool:
    """True for models that reject temperature/top_p (thinking is always on)."""
    m = model.lower()
    return any(
        tag in m
        for tag in ("opus-5", "sonnet-5", "fable-5", "mythos-5", "opus-4-8", "opus-4-7")
    )


def _anthropic_models(api_key: str) -> List[str]:
    anthropic, client = _anthropic_client(api_key)
    try:
        return sorted(m.id for m in client.models.list())
    except anthropic.AuthenticationError:
        raise ProviderError("Anthropic rejected that API key.")
    except Exception as e:
        raise ProviderError(f"Could not list Anthropic models: {e}")


# ─── errors ───────────────────────────────────────────────────────────────────


def _http_message(spec: ProviderSpec, response) -> str:
    """Turn a failed HTTP response into something a user can act on.

    Only the provider's own `error.message` is surfaced -- never the whole body,
    which can echo request headers back (including the API key).
    """
    detail = ""
    try:
        body = response.json()
        # Gemini wraps its error object in a one-element list.
        if isinstance(body, list) and body:
            body = body[0]
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                detail = err.get("message", "")
            elif isinstance(err, str):
                detail = err
            detail = detail or body.get("message", "")
    except ValueError:
        pass

    code = response.status_code
    if code in (401, 403):
        return f"{spec.label} rejected the credentials (HTTP {code}). {detail}".strip()
    if code == 404:
        return f"{spec.label}: model or endpoint not found (HTTP 404). {detail}".strip()
    if code == 429:
        return f"{spec.label} rate limit reached (HTTP 429). {detail}".strip()
    logger.error("%s error %s: %s", spec.label, code, str(detail)[:400])
    return f"{spec.label} returned HTTP {code}. {detail}".strip()


# ─── public API ───────────────────────────────────────────────────────────────


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
    """Send one completion request and return the assistant's text."""
    return chat_with_tools(
        provider_id,
        model,
        messages,
        None,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )["content"]


def chat_with_tools(
    provider_id: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> Dict[str, Any]:
    """One completion that may request tools.

    ``tools`` are flat ``{name, description, parameters}`` dicts. Returns
    ``{"content": str, "tool_calls": [{id, name, arguments}], "message": dict}``
    where ``message`` is the assistant turn to append before the tool results
    (sent back as ``{"role": "tool", "tool_call_id", "content"}``).
    """
    spec = get_spec(provider_id)
    model = (model or spec.default_model).strip()
    if not model:
        raise ProviderError(f"No model selected for {spec.label}.")
    if spec.needs_key and not api_key:
        raise ProviderError(f"{spec.label} needs an API key. Add one in Settings.")

    url = (base_url or spec.base_url or "").strip()

    if spec.kind == "anthropic":
        return _anthropic_chat(model, messages, api_key, temperature, max_tokens, tools)
    if spec.kind == "openai":
        if not url:
            raise ProviderError(f"{spec.label} needs a base URL. Add one in Settings.")
        return _openai_chat(spec, model, messages, api_key, url, temperature, max_tokens, tools)
    raise ProviderError(f"Provider kind {spec.kind!r} is not implemented.")


def list_models(
    provider_id: str, *, api_key: Optional[str] = None, base_url: Optional[str] = None
) -> List[str]:
    """Ask the provider which models it serves. Falls back to the static list."""
    spec = get_spec(provider_id)
    if not spec.can_list_models:
        return spec.models
    if spec.needs_key and not api_key:
        return spec.models

    url = (base_url or spec.base_url or "").strip()
    try:
        if spec.kind == "anthropic":
            return _anthropic_models(api_key) or spec.models
        if spec.kind == "openai":
            return _openai_models(spec, api_key, url) or spec.models
    except ProviderError:
        raise
    except Exception as e:
        raise ProviderError(f"Could not list models: {e}")
    return spec.models


def generate_image(
    provider_id: str,
    prompt: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-image-1",
    size: str = "1024x1024",
) -> Dict[str, str]:
    """Generate one image. Returns {"b64": ...} or {"url": ...}.

    Only providers whose spec sets supports_images can do this.
    """
    spec = get_spec(provider_id)
    if not spec.supports_images:
        raise ProviderError(
            f"{spec.label} cannot generate images. Switch to a provider that can "
            "(OpenAI) in Settings."
        )
    if spec.needs_key and not api_key:
        raise ProviderError(f"{spec.label} needs an API key. Add one in Settings.")

    url = f"{(base_url or spec.base_url).rstrip('/')}/images/generations"
    payload = {"model": model, "prompt": prompt, "n": 1, "size": size}
    try:
        r = requests.post(
            url, json=payload, headers=_openai_headers(spec, api_key), timeout=DEFAULT_TIMEOUT
        )
    except requests.exceptions.RequestException as e:
        raise ProviderError(f"Image request failed: {e}")

    if r.status_code != 200:
        raise ProviderError(_http_message(spec, r))

    try:
        item = r.json()["data"][0]
    except (KeyError, IndexError, ValueError):
        raise ProviderError("Unexpected image response shape.")

    if item.get("b64_json"):
        return {"b64": item["b64_json"]}
    if item.get("url"):
        return {"url": item["url"]}
    raise ProviderError("Image response contained neither data nor a URL.")


def test_connection(
    provider_id: str,
    *,
    model: str = "",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Round-trip a trivial prompt so users can verify a key before chatting."""
    spec = get_spec(provider_id)
    try:
        reply = chat(
            provider_id,
            model or spec.default_model,
            [{"role": "user", "content": "Reply with the single word: ok"}],
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=1024,
        )
    except ProviderError as e:
        return {"status": "error", "message": str(e)}
    return {
        "status": "success",
        "message": f"{spec.label} responded: {reply.strip()[:60]}",
        "model": model or spec.default_model,
    }


def supports_tools(provider_id: str) -> bool:
    """Check if a provider supports tool use/function calling.

    Args:
        provider_id: Provider ID

    Returns:
        True if provider supports tool use
    """
    get_spec(provider_id)
    # Many self-hosted models reject a `tools` payload, so leave them out.
    return provider_id in ("openai", "anthropic", "gemini", "openrouter")


def get_tools_for_provider(provider_id: str) -> List[Dict[str, Any]]:
    """Get available tools for a provider.

    Args:
        provider_id: Provider ID

    Returns:
        List of tool definitions for the provider
    """
    if not supports_tools(provider_id):
        return []

    try:
        from agent import agent_executor
        return agent_executor.get_available_tools()
    except ImportError:
        return []
