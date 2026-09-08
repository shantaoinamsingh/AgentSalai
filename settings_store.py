#!/usr/bin/env python3
"""Per-user provider settings and API credentials.

Keyed by a stable ``user_id`` cookie value, deliberately **not** by ``chat_id``:
``/chat/new`` rotates the chat id, and losing your provider configuration every
time you start a fresh conversation would be maddening.

Credential handling
-------------------
User-supplied API keys are held in process memory only. They are never written
to disk, never logged, and never returned to the browser -- the settings API
reports a masked hint (``sk-...4f2a``) and a boolean, so the UI can show that a
key is present without being able to read it back.

The trade-off is that keys are lost when the server restarts and users re-enter
them. That is the intended default: writing third-party API keys to disk in
plaintext is a far worse failure mode than re-entering one. Set
``ALLOW_CREDENTIAL_PERSISTENCE`` only if you add real encryption at rest.
"""
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

import providers

logger = logging.getLogger(__name__)

SETTINGS_TTL_SECONDS = int(os.getenv("SETTINGS_TTL_SECONDS", 30 * 24 * 60 * 60))
MAX_USERS = int(os.getenv("MAX_SETTINGS_USERS", 2000))

# Output shapes the user can request. These steer the prompt; the renderer in
# the browser then knows what to do with the result.
OUTPUT_FORMATS: Dict[str, Dict[str, str]] = {
    "auto": {
        "label": "Auto",
        "hint": "Let the assistant choose",
        "instruction": "",
    },
    "text": {
        "label": "Plain prose",
        "hint": "No tables or charts",
        "instruction": (
            "Answer in plain prose. Do not use tables, charts or code blocks "
            "unless the user explicitly asks for code."
        ),
    },
    "table": {
        "label": "Table",
        "hint": "Downloadable as CSV / Excel",
        "instruction": (
            "Present the answer as a GitHub-flavoured Markdown table wherever the "
            "data is at all tabular. Use a header row, keep one record per row, and "
            "keep cells short. Put any commentary in a short sentence before or "
            "after the table, never inside a cell."
        ),
    },
    "chart": {
        "label": "Chart",
        "hint": "Rendered as a graph",
        "instruction": (
            "Visualise the answer as a chart. Emit a fenced code block tagged "
            "`chart` containing only JSON of this shape:\n"
            '```chart\n'
            '{"type": "bar", "title": "...", '
            '"labels": ["A", "B"], '
            '"datasets": [{"label": "Series 1", "data": [1, 2]}]}\n'
            '```\n'
            "`type` may be bar, line, pie, doughnut, radar or scatter. Every "
            "dataset's `data` array must be the same length as `labels`, and must "
            "contain plain numbers. Add one or two sentences of interpretation "
            "outside the block. If you also have the underlying figures, include a "
            "Markdown table after the chart."
        ),
    },
    "report": {
        "label": "Report",
        "hint": "Structured, good for PDF",
        "instruction": (
            "Write a structured report: a short title as a Markdown H2, a one "
            "paragraph summary, then labelled sections with H3 headings. Use "
            "bullet lists and tables where they aid clarity, and finish with a "
            "'Key takeaways' list. Aim for something that reads well when printed."
        ),
    },
}

DEFAULT_SETTINGS: Dict[str, Any] = {
    "provider": providers.DEFAULT_PROVIDER,
    "model": "",
    "base_url": "",
    "temperature": float(os.getenv("LLM_TEMPERATURE", 0.7)),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", 8192)),
    "output_format": "auto",
}


def _mask(key: str) -> str:
    """A hint that identifies a key without disclosing it."""
    if not key:
        return ""
    if len(key) <= 10:
        return "•" * len(key)
    return f"{key[:3]}…{key[-4:]}"


class SettingsStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # user_id -> {"settings": {...}, "keys": {provider_id: key}, "last_seen": ts}
        self._users: Dict[str, Dict[str, Any]] = {}

    # ── internals ────────────────────────────────────────────────────────────

    def _touch(self, user_id: str) -> Dict[str, Any]:
        entry = self._users.get(user_id)
        if entry is None:
            entry = {"settings": dict(DEFAULT_SETTINGS), "keys": {}, "last_seen": time.time()}
            self._users[user_id] = entry
        entry["last_seen"] = time.time()
        return entry

    def _evict(self) -> None:
        now = time.time()
        for uid in [
            u for u, v in self._users.items() if now - v["last_seen"] > SETTINGS_TTL_SECONDS
        ]:
            self._users.pop(uid, None)
        if len(self._users) > MAX_USERS:
            ordered = sorted(self._users.items(), key=lambda kv: kv[1]["last_seen"])
            for uid, _ in ordered[: len(self._users) - MAX_USERS]:
                self._users.pop(uid, None)

    # ── reads ────────────────────────────────────────────────────────────────

    def get(self, user_id: str) -> Dict[str, Any]:
        """Effective settings, with the model defaulted for the provider."""
        with self._lock:
            self._evict()
            entry = self._touch(user_id)
            settings = dict(entry["settings"])
            keys = entry["keys"]

        try:
            spec = providers.get_spec(settings["provider"])
        except providers.ProviderError:
            settings["provider"] = providers.DEFAULT_PROVIDER
            spec = providers.get_spec(providers.DEFAULT_PROVIDER)

        if not settings.get("model"):
            settings["model"] = spec.default_model
        if not settings.get("base_url"):
            settings["base_url"] = spec.base_url
        settings["has_key"] = bool(keys.get(spec.id))
        settings["needs_key"] = spec.needs_key
        settings["supports_images"] = spec.supports_images
        return settings

    def api_key(self, user_id: str, provider_id: str) -> Optional[str]:
        with self._lock:
            entry = self._users.get(user_id)
            return (entry or {}).get("keys", {}).get(provider_id)

    def public_state(self, user_id: str) -> Dict[str, Any]:
        """Everything the settings UI renders. Contains no raw credentials."""
        settings = self.get(user_id)
        with self._lock:
            keys = dict(self._users.get(user_id, {}).get("keys", {}))
        return {
            "settings": settings,
            "providers": providers.public_catalog(),
            "output_formats": {
                fid: {"label": f["label"], "hint": f["hint"]}
                for fid, f in OUTPUT_FORMATS.items()
            },
            "saved_keys": {pid: _mask(k) for pid, k in keys.items()},
        }

    # ── writes ───────────────────────────────────────────────────────────────

    def update(self, user_id: str, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and apply a settings patch.

        An ``api_key`` in the patch is stored against the *provider it belongs
        to* and stripped from the settings dict, so it never lands anywhere that
        gets serialised back to the client.
        """
        provider_id = changes.get("provider")
        if provider_id is not None:
            try:
                providers.get_spec(provider_id)
            except providers.ProviderError as e:
                return {"status": "error", "message": str(e)}

        fmt = changes.get("output_format")
        if fmt is not None and fmt not in OUTPUT_FORMATS:
            return {"status": "error", "message": f"Unknown output format: {fmt!r}"}

        with self._lock:
            self._evict()
            entry = self._touch(user_id)
            settings = entry["settings"]

            if provider_id is not None and provider_id != settings.get("provider"):
                # Switching provider: clear model/base_url so the new provider's
                # defaults apply instead of a stale value from the old one.
                settings["provider"] = provider_id
                settings["model"] = ""
                settings["base_url"] = ""

            if "model" in changes:
                settings["model"] = str(changes["model"] or "").strip()
            if "base_url" in changes:
                settings["base_url"] = str(changes["base_url"] or "").strip()
            if "output_format" in changes:
                settings["output_format"] = changes["output_format"]

            if "temperature" in changes:
                try:
                    settings["temperature"] = max(0.0, min(2.0, float(changes["temperature"])))
                except (TypeError, ValueError):
                    return {"status": "error", "message": "temperature must be a number"}
            if "max_tokens" in changes:
                try:
                    settings["max_tokens"] = max(256, min(128000, int(changes["max_tokens"])))
                except (TypeError, ValueError):
                    return {"status": "error", "message": "max_tokens must be a whole number"}

            if "api_key" in changes:
                target = provider_id or settings["provider"]
                raw = str(changes["api_key"] or "").strip()
                if raw:
                    entry["keys"][target] = raw
                    logger.info("Stored an API key for provider %s (in memory only)", target)
                else:
                    entry["keys"].pop(target, None)

        return {"status": "success", **self.public_state(user_id)}

    def clear_key(self, user_id: str, provider_id: str) -> Dict[str, Any]:
        with self._lock:
            entry = self._users.get(user_id)
            removed = bool(entry and entry["keys"].pop(provider_id, None))
        return {
            "status": "success",
            "removed": removed,
            "message": f"Key for {provider_id} removed" if removed else "No key stored",
        }

    def reset(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            self._users.pop(user_id, None)
        return {"status": "success", "message": "Settings reset to defaults"}

    # ── prompt steering ──────────────────────────────────────────────────────

    def format_instruction(self, user_id: str) -> str:
        return OUTPUT_FORMATS.get(self.get(user_id)["output_format"], {}).get("instruction", "")


settings_store = SettingsStore()
