#!/usr/bin/env python3
"""Server-side per-chat state: conversation history and session-scoped files.

Why this exists instead of using the Flask session
--------------------------------------------------
Two independent problems make the cookie session unusable here:

1. Flask-SocketIO handlers can *read* the session (it is loaded from the
   WebSocket handshake cookie) but writes never reach the browser, because
   there is no HTTP response to set a cookie on. The previous code appended
   chat turns to ``session['io_ChatHistory']`` inside a socket handler, so
   every write was silently discarded and the assistant had no memory.

2. Cookies cap at ~4 KB. Per-chat file text is far larger than that.

So the cookie holds only a small opaque ``chat_id`` (written by an ordinary
HTTP route, where cookies work) and everything else lives here, in process
memory, keyed by that id.

Scope: this is an in-process store, correct for the single-process
``socketio.run`` deployment this app uses. Running multiple workers would need
a shared backend such as Redis.
"""
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Drop a chat this long after its last activity.
CHAT_TTL_SECONDS = int(os.getenv("CHAT_TTL_SECONDS" or 8 * 60 * 60))

# Hard cap on tracked chats, so an unbounded number of visitors cannot
# exhaust memory. Least-recently-used chats are evicted first.
MAX_CHATS = int(os.getenv("MAX_CHATS", 500))

# Conversation turns retained per chat (user + assistant messages).
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", 20))

# Per-chat file text is injected into the prompt verbatim rather than being
# embedded and retrieved: the user just attached the file and expects the model
# to see all of it, so retrieval could only lose information. These caps keep
# the prompt inside the model's context window.
MAX_CONTEXT_CHARS_PER_FILE = int(os.getenv("MAX_CONTEXT_CHARS_PER_FILE", 40_000))
MAX_CONTEXT_CHARS_TOTAL = int(os.getenv("MAX_CONTEXT_CHARS_TOTAL", 60_000))
MAX_CONTEXT_FILES = int(os.getenv("MAX_CONTEXT_FILES", 10))


class ChatStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chats: Dict[str, Dict[str, Any]] = {}

    # ── internals ────────────────────────────────────────────────────────────

    def _new_chat(self) -> Dict[str, Any]:
        now = time.time()
        return {"history": [], "context": {}, "created_at": now, "last_seen": now}

    def _touch(self, chat_id: str) -> Dict[str, Any]:
        """Fetch (creating if needed) a chat and mark it active. Assumes lock."""
        chat = self._chats.get(chat_id)
        if chat is None:
            chat = self._new_chat()
            self._chats[chat_id] = chat
        chat["last_seen"] = time.time()
        return chat

    def _evict(self) -> None:
        """Expire idle chats and enforce MAX_CHATS. Assumes lock held."""
        now = time.time()
        for cid in [c for c, v in self._chats.items() if now - v["last_seen"] > CHAT_TTL_SECONDS]:
            self._chats.pop(cid, None)
            logger.info("Expired idle chat %s", cid[:8])

        if len(self._chats) > MAX_CHATS:
            ordered = sorted(self._chats.items(), key=lambda kv: kv[1]["last_seen"])
            for cid, _ in ordered[: len(self._chats) - MAX_CHATS]:
                self._chats.pop(cid, None)
                logger.info("Evicted least-recently-used chat %s", cid[:8])

    # ── history ──────────────────────────────────────────────────────────────

    def get_history(self, chat_id: str) -> List[Dict[str, str]]:
        with self._lock:
            self._evict()
            return list(self._touch(chat_id)["history"])

    def append_message(self, chat_id: str, role: str, content: str) -> None:
        """Record one turn, trimming oldest turns past MAX_HISTORY_MESSAGES."""
        with self._lock:
            chat = self._touch(chat_id)
            chat["history"].append({"role": role, "content": content})
            if len(chat["history"]) > MAX_HISTORY_MESSAGES:
                del chat["history"][: len(chat["history"]) - MAX_HISTORY_MESSAGES]

    def clear_history(self, chat_id: str) -> None:
        with self._lock:
            self._touch(chat_id)["history"] = []

    # ── per-chat context files ───────────────────────────────────────────────

    def add_context(self, chat_id: str, filename: str, text: str) -> Dict[str, Any]:
        """Attach a file's text to this chat only.

        Re-attaching the same filename replaces the earlier copy.
        """
        text = (text or "").strip()
        if not text:
            return {"status": "error", "message": "No extractable text found in file"}

        truncated = False
        if len(text) > MAX_CONTEXT_CHARS_PER_FILE:
            text = text[:MAX_CONTEXT_CHARS_PER_FILE]
            truncated = True

        with self._lock:
            self._evict()
            chat = self._touch(chat_id)
            context: Dict[str, Dict[str, Any]] = chat["context"]

            if filename not in context and len(context) >= MAX_CONTEXT_FILES:
                return {
                    "status": "error",
                    "message": f"This chat already has {MAX_CONTEXT_FILES} files attached. "
                    "Remove one first.",
                }

            other_chars = sum(v["chars"] for k, v in context.items() if k != filename)
            if other_chars + len(text) > MAX_CONTEXT_CHARS_TOTAL:
                remaining = MAX_CONTEXT_CHARS_TOTAL - other_chars
                if remaining < 1000:
                    return {
                        "status": "error",
                        "message": "This chat's context is full. Remove a file first.",
                    }
                text = text[:remaining]
                truncated = True

            context[filename] = {
                "text": text,
                "chars": len(text),
                "truncated": truncated,
                "added_at": time.time(),
            }

        message = f"{filename} attached to this chat ({len(text):,} chars)"
        if truncated:
            message += " — truncated to fit the context limit"
        logger.info("[chat %s] %s", chat_id[:8], message)
        return {
            "status": "success",
            "filename": filename,
            "chars": len(text),
            "truncated": truncated,
            "message": message,
        }

    def list_context(self, chat_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            chat = self._touch(chat_id)
            return [
                {
                    "filename": name,
                    "chars": entry["chars"],
                    "truncated": entry["truncated"],
                    "added_at": entry["added_at"],
                }
                for name, entry in sorted(chat["context"].items(), key=lambda kv: kv[1]["added_at"])
            ]

    def remove_context(self, chat_id: str, filename: str) -> Dict[str, Any]:
        with self._lock:
            chat = self._touch(chat_id)
            if filename not in chat["context"]:
                return {"status": "error", "message": f"{filename} is not attached to this chat"}
            chat["context"].pop(filename)
        return {"status": "success", "message": f"Removed {filename} from this chat"}

    def clear_context(self, chat_id: str) -> Dict[str, Any]:
        with self._lock:
            chat = self._touch(chat_id)
            count = len(chat["context"])
            chat["context"] = {}
        return {"status": "success", "removed": count, "message": f"Detached {count} file(s)"}

    def build_context_text(self, chat_id: str) -> str:
        """Render attached files for injection into the system prompt."""
        with self._lock:
            chat = self._chats.get(chat_id)
            if not chat or not chat["context"]:
                return ""
            entries = sorted(chat["context"].items(), key=lambda kv: kv[1]["added_at"])
            return "\n\n".join(f"[File: {name}]\n{entry['text']}" for name, entry in entries)

    # ── whole-chat lifecycle ─────────────────────────────────────────────────

    def reset_chat(self, chat_id: str) -> None:
        """Forget a chat entirely: history and attached files."""
        with self._lock:
            self._chats.pop(chat_id, None)
        logger.info("Reset chat %s", chat_id[:8])

    def stats(self, chat_id: str) -> Dict[str, Any]:
        with self._lock:
            chat = self._touch(chat_id)
            files = chat["context"]
            return {
                "messages": len(chat["history"]),
                "files": len(files),
                "context_chars": sum(v["chars"] for v in files.values()),
                "max_context_chars": MAX_CONTEXT_CHARS_TOTAL,
                "max_files": MAX_CONTEXT_FILES,
                "active_chats": len(self._chats),
            }


chat_store = ChatStore()
