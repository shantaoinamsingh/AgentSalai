#!/usr/bin/env python3
"""Pluggable LLM providers.

The app started life hard-wired to one Azure OpenAI deployment behind dentsu's
APIM gateway. That gateway is heavily restricted -- it rejects everything that
is not a GPT-4/o-series chat model, which is why embeddings run locally and why
image generation is impossible through it:

    403 {"error": {"message": "Model not allowed. You can only use o1, o3,
         o3-deep-research, or GPT-4 models.", "code": "model_not_allowed"}}

So users can now pick a provider and supply their own credentials. The dentsu
gateway stays the default (it needs no user key); anything else unlocks the
models -- and the capabilities -- that the gateway blocks.

Every provider takes and returns the same shapes:

    chat(messages, ...) -> str            messages are OpenAI-style dicts
    list_models(...)    -> list[str]
    generate_image(...) -> {"b64" | "url"}

Credentials are passed in per call. Nothing in this module reads or writes them
from disk; see settings_store.py for how they are held.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = int(os.getenv("LLM_TIMEOUT_SECONDS" or 120))


class ProviderError(RuntimeError):
    """A provider call failed in a way worth showing the user."""


@dataclass
class ProviderSpec:
    id: str
    label: str
    kind: str  # azure_apim | openai | anthropic
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
    "dentsu": ProviderSpec(
        id="dentsu",
        label="dentsu Azure OpenAI",
        kind="azure_apim",
        needs_key=False,
        description="Shared internal gateway. No key needed, but it only allows "
        "GPT-4 and o-series chat models (no embeddings, no image generation).",
        models=["GPT4o128k", "o1", "o3"],
        default_model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "GPT4o128k"),
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    ),
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

DEFAULT_PROVIDER = "dentsu"


def get_spec(provider_id: str) -> ProviderSpec:
    spec = PROVIDERS.get(provider_id)
    if spec is None:
        raise ProviderError(f"Unknown provider: {provider_id!r}")
    return spec


def public_catalog() -> List[Dict[str, Any]]:
    return [p.public() for p in PROVIDERS.values()]


# ─── message shaping ──────────────────────────────────────────────────────────


def _split_system(messages: List[Dict[str, str]]) -> tuple:
    """Separate system text from the turn list (Anthropic keeps them apart)."""
    system_chunks = [m["content"] for m in messages if m.get("role") == "system"]
    turns = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant")
    ]
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
    messages: List[Dict[str, str]],
    api_key: Optional[str],
    base_url: str,
    temperature: float,
    max_tokens: int,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(
            url, json=payload, headers=_openai_headers(spec, api_key), timeout=DEFAULT_TIMEOUT
        )
    except requests.exceptions.Timeout:
        raise ProviderError(f"{spec.label} timed out after {DEFAULT_TIMEOUT}s.")
    except requests.exceptions.RequestException as e:
        raise ProviderError(f"Could not reach {spec.label}: {e}")

    if r.status_code != 200:
        raise ProviderError(_http_message(spec, r))

    try:
        return r.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError):
        raise ProviderError(f"{spec.label} returned an unexpected response shape.")


def _openai_models(spec: ProviderSpec, api_key: Optional[str], base_url: str) -> List[str]:
    url = f"{base_url.rstrip('/')}/models"
    r = requests.get(url, headers=_openai_headers(spec, api_key), timeout=30)
    if r.status_code != 200:
        raise ProviderError(_http_message(spec, r))
    data = r.json().get("data", [])
    ids = sorted(m["id"] for m in data if isinstance(m, dict) and m.get("id"))
    return ids


# ─── dentsu APIM (Azure OpenAI behind a gateway) ──────────────────────────────


def _dentsu_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    server_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint or not server_key:
        raise ProviderError(
            "The dentsu gateway is not configured on the server. "
            "Pick another provider, or set AZURE_OPENAI_* in .env."
        )

    url = f"{endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}"
    headers = {
        "x-brand": os.getenv("API_BRAND", "dentsu"),
        "x-service-line": os.getenv("API_SERVICE_LINE", "Functions"),
        "x-project": os.getenv("API_PROJECT", "test"),
        "Ocp-Apim-Subscription-Key": server_key,
        "api-version": os.getenv("API_GATEWAY_VERSION", "v15"),
        "Content-Type": "application/json",
    }
    # This gateway wants max_completion_tokens, not max_tokens.
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
    except requests.exceptions.Timeout:
        raise ProviderError(f"The dentsu gateway timed out after {DEFAULT_TIMEOUT}s.")
    except requests.exceptions.RequestException as e:
        raise ProviderError(f"Could not reach the dentsu gateway: {e}")

    if r.status_code != 200:
        raise ProviderError(_http_message(PROVIDERS["dentsu"], r))
    try:
        return r.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError):
        raise ProviderError("The dentsu gateway returned an unexpected response shape.")


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
) -> str:
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
    if not parts:
        raise ProviderError("Claude returned no text content.")
    return "".join(parts)


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
        base = f"{spec.label} rejected the credentials"
        if spec.id == "dentsu":
            base = f"{spec.label} refused the request"
        return f"{base} (HTTP {code}). {detail}".strip()
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
    spec = get_spec(provider_id)
    model = (model or spec.default_model).strip()
    if not model:
        raise ProviderError(f"No model selected for {spec.label}.")
    if spec.needs_key and not api_key:
        raise ProviderError(f"{spec.label} needs an API key. Add one in Settings.")

    url = (base_url or spec.base_url or "").strip()

    if spec.kind == "azure_apim":
        return _dentsu_chat(model, messages, temperature, max_tokens)
    if spec.kind == "anthropic":
        return _anthropic_chat(model, messages, api_key, temperature, max_tokens)
    if spec.kind == "openai":
        if not url:
            raise ProviderError(f"{spec.label} needs a base URL. Add one in Settings.")
        return _openai_chat(spec, model, messages, api_key, url, temperature, max_tokens)
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

    Only providers whose spec sets supports_images can do this -- notably not
    the dentsu gateway, which blocks non-chat models outright.
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
