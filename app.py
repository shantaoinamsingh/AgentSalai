#!/usr/bin/env python3
"""Salai - Flask + SocketIO chat with pluggable LLM providers.

Two tiers of context, deliberately separate:

* **Organisation context** (``/admin/*``, requires ``ADMIN_API_KEY``) -- files
  admins curate for everyone. Persisted in Chroma, embedded, and retrieved by
  semantic similarity. Lives in ``knowledge_base.py``.

* **Chat context** (``/chat/*``, no admin key) -- files a user attaches to
  their own conversation. Held in memory, injected into the prompt verbatim,
  and discarded when the session ends. Lives in ``chat_store.py``.

Two session identities, also deliberately separate:

* ``chat_id`` -- rotates on "New chat"; keys history and attachments.
* ``user_id`` -- stable for the browser; keys provider settings and API keys,
  which must survive starting a new conversation.
"""
import base64
import copy
import functools
import hmac
import logging
import os
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from flask import Flask, Response, jsonify, render_template, request, session
from flask_socketio import SocketIO, emit

# Load environment before importing modules that read it at import time.
from dotenv import load_dotenv

load_dotenv()

import exporters  # noqa: E402
import providers  # noqa: E402
import chat_store as chat_store_limits  # noqa: E402
from chat_store import chat_store  # noqa: E402
from knowledge_base import SUPPORTED_EXTENSIONS, kb  # noqa: E402
from settings_store import OUTPUT_FORMATS, settings_store  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from agent import agent_executor, run_sync as run_agent_sync
    from browser_tools import browser_tools as browser_manager

    HAS_BROWSER_TOOLS = bool(agent_executor.get_available_tools())
except ImportError:
    HAS_BROWSER_TOOLS = False
if not HAS_BROWSER_TOOLS:
    logger.info("Browser tools not available (install playwright to enable)")

# Visible by default so a local user can watch the agent work.
BROWSER_HEADLESS = os.getenv("BROWSER_HEADLESS", "false").lower() in ("1", "true", "yes")
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", 15))
# How long to wait for the user to answer a site-access prompt before declining.
PERMISSION_TIMEOUT = int(os.getenv("BROWSER_PERMISSION_TIMEOUT", 120))
MAX_TOOL_RESULT_CHARS = 4000

# Only these are needed to boot. Provider credentials are not required since
# users can supply their own key for OpenAI/Anthropic/OpenRouter in Settings.
REQUIRED_VARS = ["SECRET_KEY", "ADMIN_API_KEY"]

missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing)}. "
        "Copy .env.example to .env and fill it in."
    )

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 50 * 1024 * 1024))

SYSTEM_BASE = (
    "You are Salai, a helpful AI assistant with web browsing capabilities. "
    "Answer clearly and concisely, using Markdown where it helps.\n"
    # Retrieval runs per turn, so a follow-up question may carry no context
    # even though an earlier turn was grounded in real documents. Without this
    # the model disclaims its own previous answer ("I don't actually have
    # access to those documents"), which reads as a contradiction to the user.
    "Context is supplied per message, so documents quoted earlier in this "
    "conversation may not be repeated below. Trust your earlier answers and "
    "never claim you lack access to material you have already been shown.\n"
)

# Add browser tools info if available
if HAS_BROWSER_TOOLS:
    SYSTEM_BASE += """
## Browser Automation Capabilities

You have access to browser automation tools. When users ask you to:
- Search for information online
- Check if a website is working
- Extract data from websites
- Take screenshots
- Fill and submit forms

You should use the browser tools to complete these tasks autonomously instead of giving manual instructions.

Available tools: open_browser, navigate, click_element, fill_input, take_screenshot, execute_js, close_browser

Always use these tools when appropriate rather than telling users to do things manually.
Call open_browser first. The browser stays open between messages, so leave it open
unless the user asks you to close it. To search the web, navigate straight to
https://www.bing.com/search?q=your+query (Google and DuckDuckGo block automated
browsers with a CAPTCHA) rather than filling a search box. The navigate result
includes the page text, so read results from it.
Any site can be requested: the app asks the user before opening a site that is not
yet trusted, so never refuse up front. If the user declines, respect that and do not retry.
"""

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# Jinja caches templates whenever debug is off, so UI edits would need a
# restart to appear. Opt in without turning on the debugger.
app.config["TEMPLATES_AUTO_RELOAD"] = (
    os.getenv("TEMPLATES_AUTO_RELOAD", "true").lower() == "true"
)
# Enable when serving over HTTPS (the cookie only carries an opaque chat id).
app.config["SESSION_COOKIE_SECURE"] = os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"

socketio = SocketIO(app, max_http_buffer_size=MAX_UPLOAD_BYTES)

# Upload jobs, so a slow document does not block the HTTP response.
_upload_jobs: Dict[str, Dict[str, Any]] = {}
_upload_lock = threading.Lock()
UPLOAD_JOB_TTL_SECONDS = int(os.getenv("UPLOAD_JOB_TTL_SECONDS", 3600))
MAX_UPLOAD_JOBS = int(os.getenv("MAX_UPLOAD_JOBS", 200))


# ─── auth ─────────────────────────────────────────────────────────────────────


def _provided_api_key() -> Optional[str]:
    """Read the admin key from a header only.

    Query parameters are deliberately not accepted: they end up in access
    logs, browser history and Referer headers, and this key authorises writes
    to the organisation-wide context.
    """
    header = request.headers.get("X-API-Key")
    if header:
        return header.strip()
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def require_admin(view):
    """Gate a view on ADMIN_API_KEY, supplied via the X-API-Key header."""

    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        provided = _provided_api_key()
        if not provided:
            return jsonify(
                status="error",
                message="Admin API key required in the X-API-Key header",
            ), 401
        if not hmac.compare_digest(provided, os.getenv("ADMIN_API_KEY", "")):
            logger.warning("Rejected admin request to %s from %s", request.path, request.remote_addr)
            return jsonify(status="error", message="Invalid admin API key"), 403
        return view(*args, **kwargs)

    return wrapper


def current_chat_id() -> str:
    """Stable id for this browser's conversation.

    Written by an HTTP route (where cookies work) and merely read inside
    SocketIO handlers, which cannot set cookies. Falls back to the socket id
    if a client somehow connects without loading the page first.
    """
    chat_id = session.get("chat_id")
    if chat_id:
        return chat_id
    sid = getattr(request, "sid", None)
    if sid:
        return f"sid:{sid}"
    chat_id = uuid.uuid4().hex
    session["chat_id"] = chat_id
    return chat_id


def current_user_id() -> str:
    """Stable id for this browser, independent of the current conversation.

    Provider settings and API keys hang off this rather than ``chat_id`` so that
    starting a new chat does not wipe the user's configuration.
    """
    user_id = session.get("user_id")
    if user_id:
        return user_id
    sid = getattr(request, "sid", None)
    if sid:
        return f"sid:{sid}"
    user_id = uuid.uuid4().hex
    session["user_id"] = user_id
    return user_id


# ─── prompt assembly ──────────────────────────────────────────────────────────


def build_messages(
    chat_id: str, user_input: str, user_id: Optional[str] = None
) -> Tuple[list, Dict[str, Any]]:
    """Compose the request payload for one turn.

    Context goes in the system message *only*. The user's message stays
    verbatim -- the previous implementation wrapped retrieved chunks into the
    user turn *and* repeated them in the system message, which doubled token
    cost and left the model unable to tell instructions from data.
    """
    meta: Dict[str, Any] = {
        "kb_status": "skipped",
        "kb_sources": [],
        "chat_files": 0,
        "output_format": "auto",
    }

    system_parts = [SYSTEM_BASE]

    # The chosen output shape is an instruction, so it belongs with the other
    # system-level directives rather than tacked onto the user's message.
    if user_id:
        fmt = settings_store.get(user_id)["output_format"]
        meta["output_format"] = fmt
        instruction = OUTPUT_FORMATS.get(fmt, {}).get("instruction", "")
        if instruction:
            system_parts.append("## Requested output format\n" + instruction)

    result = kb.query(user_input)
    meta["kb_status"] = result["status"]
    if result["status"] == "ok":
        meta["kb_sources"] = result["sources"]
        system_parts.append(
            "## Organisation knowledge base\n"
            "Excerpts retrieved for this question. Prefer them over general knowledge "
            "and cite the source filename when you use one.\n\n" + result["context"]
        )
    elif result["status"] == "stale":
        logger.warning("Knowledge base index is stale; run: python manage_kb.py reindex")

    chat_context = chat_store.build_context_text(chat_id)
    if chat_context:
        attached = chat_store.list_context(chat_id)
        meta["chat_files"] = len(attached)
        system_parts.append(
            "## Files attached to this conversation\n"
            "The user attached these to this chat. They are authoritative for "
            "questions about their own documents.\n\n" + chat_context
        )

    if len(system_parts) > 1:
        system_parts.append(
            "If the material above does not cover the question, say so briefly and "
            "then answer from general knowledge."
        )

    messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
    messages.extend(chat_store.get_history(chat_id))

    images = chat_store.get_images(chat_id)
    if images:
        # OpenAI-style parts; providers.py converts them for Anthropic.
        content: Any = [{"type": "text", "text": user_input}] + [
            {"type": "image_url", "image_url": {"url": f"data:{i['media_type']};base64,{i['data']}"}}
            for i in images
        ]
    else:
        content = user_input
    messages.append({"role": "user", "content": content})
    meta["chat_images"] = len(images)
    return messages, meta


def get_answer(user_input: str, chat_id: str, user_id: str, ask_permission=None) -> str:
    """Run one conversation turn and persist it to the chat's history.

    With browser tools support, the agent can browse websites, take screenshots,
    fill forms, and execute JavaScript when needed.
    """
    user_input = (user_input or "").strip()
    if not user_input:
        return "Please enter a question."

    messages, meta = build_messages(chat_id, user_input, user_id)
    settings = settings_store.get(user_id)
    provider_id = settings["provider"]

    logger.info(
        "[chat %s] provider=%s model=%s format=%s kb=%s sources=%s attached=%d history=%d",
        chat_id[:8],
        provider_id,
        settings["model"],
        meta["output_format"],
        meta["kb_status"],
        meta["kb_sources"] or "-",
        meta["chat_files"],
        len(messages) - 2,
    )

    chat_kwargs = dict(
        api_key=settings_store.api_key(user_id, provider_id),
        base_url=settings["base_url"],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
    )
    try:
        if HAS_BROWSER_TOOLS and providers.supports_tools(provider_id):
            answer = _run_tool_loop(
                provider_id, settings["model"], messages, chat_id, chat_kwargs, ask_permission
            )
        else:
            answer = providers.chat(provider_id, settings["model"], messages, **chat_kwargs)
    except providers.ProviderError as e:
        # These messages are written to be shown to the user, and the provider
        # layer keeps raw response bodies (which can echo the API key) out.
        logger.error("[chat %s] provider %s failed: %s", chat_id[:8], provider_id, e)
        return f"⚠️ {e}"

    # Record both sides of the turn. The old code stored only the assistant
    # message, which produced a malformed, assistant-only "conversation".
    chat_store.append_message(chat_id, "user", user_input)
    chat_store.append_message(chat_id, "assistant", answer)
    return answer


def _browser_tool_defs() -> list:
    """Tool definitions for the LLM, minus the params the app fills in itself."""
    defs = []
    for tool in agent_executor.get_available_tools():
        tool = copy.deepcopy(tool)
        params = tool.setdefault("parameters", {"type": "object", "properties": {}})
        for hidden in ("session_id", "headless"):
            params.get("properties", {}).pop(hidden, None)
            if hidden in params.get("required", []):
                params["required"].remove(hidden)
        defs.append(tool)
    return defs


def _check_site_access(url: str, chat_id: str, ask_permission) -> Optional[str]:
    """Ask the user before visiting an untrusted site. Returns a refusal, or None to proceed."""
    if not browser_manager.is_safe_url(url) or browser_manager.is_trusted(url, chat_id):
        return None  # unsafe URLs are refused by navigate itself
    domain = browser_manager.domain_of(url)
    decision = ask_permission(domain, url) if ask_permission else "deny"
    logger.info("[chat %s] site access %s -> %s", chat_id[:8], domain, decision)
    if decision == "always":
        browser_manager.trust_domain(domain)
    elif decision == "once":
        browser_manager.approve_domain(chat_id, domain)
    else:
        return (
            f"The user declined access to {domain}. Do not retry this site; "
            "try another source or tell the user what you could not do."
        )
    return None


def _run_tool_loop(
    provider_id: str, model: str, messages: list, chat_id: str, chat_kwargs: dict, ask_permission=None
) -> str:
    """Let the model call browser tools until it produces a final answer."""
    tools = _browser_tool_defs()
    for _ in range(MAX_TOOL_ROUNDS):
        turn = providers.chat_with_tools(provider_id, model, messages, tools, **chat_kwargs)
        if not turn["tool_calls"]:
            return turn["content"] or "(no response)"

        messages.append(turn["message"])
        for call in turn["tool_calls"]:
            args = dict(call["arguments"] or {})
            # One browser per conversation; the model never picks the session.
            args["session_id"] = chat_id
            if call["name"] == "open_browser":
                args["headless"] = BROWSER_HEADLESS
            logger.info("[chat %s] tool %s %s", chat_id[:8], call["name"], {k: v for k, v in args.items() if k != "session_id"})
            try:
                refusal = None
                if call["name"] == "navigate" and args.get("url"):
                    refusal = _check_site_access(str(args["url"]), chat_id, ask_permission)
                result = refusal or run_agent_sync(
                    agent_executor.execute_tool(call["name"], args, chat_id)
                )
            except Exception as e:
                logger.exception("[chat %s] tool %s failed", chat_id[:8], call["name"])
                result = f"Error: {e}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": str(result)[:MAX_TOOL_RESULT_CHARS],
                }
            )

    return f"I stopped after {MAX_TOOL_ROUNDS} browser steps without finishing. Try a more specific request."


# ─── page ─────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    # Establish both ids here, where setting a cookie actually works.
    if "chat_id" not in session:
        session["chat_id"] = uuid.uuid4().hex
        session.permanent = False
    if "user_id" not in session:
        session["user_id"] = uuid.uuid4().hex
    return render_template("index.html")


@app.errorhandler(413)
def too_large(_e):
    return jsonify(
        status="error",
        message=f"File exceeds the {MAX_UPLOAD_BYTES / 1e6:.0f} MB limit",
    ), 413


# ─── shared upload helper ─────────────────────────────────────────────────────


def _validated_upload() -> Tuple[Optional[Any], Optional[Tuple[Any, int]]]:
    """Pull `file` off the request and reject bad ones early.

    Returns ``(file, None)`` on success or ``(None, (response, status))``.
    Extension is checked *before* the upload is written to disk.
    """
    if "file" not in request.files:
        return None, (jsonify(status="error", message="No file provided in the request"), 400)

    file = request.files["file"]
    if not file.filename:
        return None, (jsonify(status="error", message="No file selected"), 400)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return None, (
            jsonify(
                status="error",
                message=f"Unsupported file type {ext or '(none)'}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            ),
            400,
        )
    return file, None


def _spool_to_temp(file) -> str:
    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp)
        return tmp.name


# ─── organisation context (admin) ─────────────────────────────────────────────


def _prune_jobs() -> None:
    """Expire finished jobs. Assumes _upload_lock held."""
    now = time.time()
    for jid in [
        j
        for j, v in _upload_jobs.items()
        if v.get("completed_at") and now - v["completed_at"] > UPLOAD_JOB_TTL_SECONDS
    ]:
        _upload_jobs.pop(jid, None)
    if len(_upload_jobs) > MAX_UPLOAD_JOBS:
        for jid, _ in sorted(_upload_jobs.items(), key=lambda kv: kv[1]["started_at"])[
            : len(_upload_jobs) - MAX_UPLOAD_JOBS
        ]:
            _upload_jobs.pop(jid, None)


def _process_upload(job_id: str, tmp_path: str, filename: str, api_key: str) -> None:
    try:
        result = kb.ingest_uploaded_file(tmp_path, filename, api_key)
        status = result.get("status", "error")
    except PermissionError:
        result, status = {"status": "error", "message": "Invalid admin API key"}, "error"
    except Exception as e:
        logger.exception("[upload-job %s] failed", job_id)
        result, status = {"status": "error", "message": str(e)}, "error"
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    with _upload_lock:
        if job_id in _upload_jobs:
            _upload_jobs[job_id].update(
                status=status, result=result, completed_at=time.time()
            )
    logger.info("[upload-job %s] %s: %s", job_id, status, result.get("message", ""))


@app.route("/admin/upload_context", methods=["POST"])
@require_admin
def upload_context():
    """Add a file to the organisation-wide knowledge base.

    Indexing runs in the background; poll ``/admin/upload_status/<job_id>``.
    """
    file, error = _validated_upload()
    if error:
        return error

    tmp_path = _spool_to_temp(file)
    job_id = uuid.uuid4().hex

    with _upload_lock:
        _prune_jobs()
        _upload_jobs[job_id] = {
            "status": "processing",
            "filename": file.filename,
            "started_at": time.time(),
            "completed_at": None,
            "result": None,
        }

    threading.Thread(
        target=_process_upload,
        args=(job_id, tmp_path, file.filename, _provided_api_key()),
        daemon=True,
    ).start()

    return jsonify(
        status="processing",
        job_id=job_id,
        filename=file.filename,
        message="Upload accepted; indexing in the background.",
    ), 202


@app.route("/admin/upload_status/<job_id>", methods=["GET"])
@require_admin
def upload_status(job_id):
    with _upload_lock:
        job = _upload_jobs.get(job_id)
        snapshot = dict(job) if job else None
    if snapshot is None:
        return jsonify(status="error", message="Unknown or expired job_id"), 404
    return jsonify(job_id=job_id, **snapshot)


@app.route("/admin/list_context", methods=["GET"])
@require_admin
def list_context():
    docs = kb.list_indexed_documents()
    return jsonify(status="success", documents=docs, count=len(docs))


@app.route("/admin/delete_context", methods=["DELETE"])
@require_admin
def delete_context():
    source = request.args.get("source")
    if not source:
        return jsonify(status="error", message="Query parameter 'source' is required"), 400
    result = kb.remove_document(source)
    return jsonify(result), 200 if result.get("status") == "success" else 400


@app.route("/admin/context_status", methods=["GET"])
@require_admin
def context_status():
    return jsonify(status="success", **kb.get_context_status())


@app.route("/admin/kb_status", methods=["GET"])
@require_admin
def kb_status():
    status = kb.get_context_status()
    if not kb.ready:
        message = (
            "Index is stale (built with a different embedding model). "
            "Run: python manage_kb.py reindex"
            if kb.stale
            else f"Knowledge base unavailable: {kb.init_error}"
        )
        return jsonify(status="error", message=message, **status)
    return jsonify(
        status="ready",
        message=f"Ready with {status['total_chunks']} chunks from {status['total_files']} file(s)",
        **status,
    )


@app.route("/admin/index_kb", methods=["POST"])
@require_admin
def index_kb():
    """Bulk-index a local folder into the organisation knowledge base."""
    folder = request.args.get("folder", "")
    if not folder:
        return jsonify(status="error", message="Query parameter 'folder' is required"), 400
    try:
        result = kb.index_documents(folder, _provided_api_key())
    except PermissionError:
        return jsonify(status="error", message="Invalid admin API key"), 403
    except Exception as e:
        logger.exception("Bulk indexing failed")
        return jsonify(status="error", message=str(e)), 500
    return jsonify(result), 200 if result.get("status") == "success" else 400


@app.route("/admin/prune_uploads", methods=["POST"])
@require_admin
def prune_uploads():
    """Delete stored files no longer referenced by the index."""
    return jsonify(status="success", **kb.prune_orphans())


# ─── chat context (current conversation only) ─────────────────────────────────


@app.route("/chat/upload_context", methods=["POST"])
def chat_upload_context():
    """Attach a file to the current chat only.

    No admin key: the file is scoped to this browser session, is never written
    to the shared index, and disappears when the session ends. Text is
    extracted synchronously -- per-chat files are prompt-sized, so there is no
    embedding work to wait on.
    """
    chat_id = current_chat_id()
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify(status="error", message="No file provided in the request"), 400
    filename = os.path.basename(file.filename)

    tmp_path = _spool_to_temp(file)
    try:
        image_type = _vision_media_type(tmp_path)
        if image_type:
            with open(tmp_path, "rb") as f:
                raw = f.read()
            result = chat_store.add_image(
                chat_id, filename, image_type, base64.b64encode(raw).decode("ascii"), len(raw)
            )
        else:
            text = _extract_chat_file(tmp_path, filename, file.mimetype)
            result = chat_store.add_context(chat_id, filename, text)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return jsonify(result), 200 if result["status"] == "success" else 400


# Formats all vision providers accept (OpenAI, Anthropic, Gemini).
_VISION_TYPES = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
}


def _vision_media_type(path: str) -> Optional[str]:
    """Detect a model-viewable image from its bytes, not its (possibly missing) name."""
    with open(path, "rb") as f:
        head = f.read(16)
    for magic, media_type in _VISION_TYPES.items():
        if head.startswith(magic):
            return media_type
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


def _extract_chat_file(path: str, filename: str, mimetype: str) -> str:
    """Best-effort text for any file type. Never empty: unreadable files get a description."""
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    if ext in SUPPORTED_EXTENSIONS:
        text = kb.extract_text(path)
    elif ext in (".xlsx", ".xlsm"):
        text = _extract_workbook(path)
    else:
        text = _read_if_text(path)

    if text.strip():
        return text
    size = os.path.getsize(path)
    kind = mimetype or "unknown type"
    note = " It may be a scanned document." if ext == ".pdf" else ""
    return (
        f"(The user attached {filename}, {size:,} bytes, {kind}. Its contents could not be "
        f"read as text.{note} Tell the user if you need it in another format.)"
    )


def _read_if_text(path: str) -> str:
    """Decode files that are really text (code, JSON, XML, logs...) regardless of extension."""
    with open(path, "rb") as f:
        raw = f.read(chat_store_limits.MAX_CONTEXT_CHARS_PER_FILE * 4)
    if b"\x00" in raw[:8192]:
        return ""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _extract_workbook(path: str) -> str:
    from openpyxl import load_workbook

    try:
        wb = load_workbook(path, read_only=True, data_only=True)
    except Exception as e:
        logger.warning("Could not open workbook %s: %s", path, e)
        return ""
    parts = []
    for ws in wb.worksheets:
        rows = [
            "\t".join("" if v is None else str(v) for v in row)
            for row in ws.iter_rows(values_only=True)
            if any(v is not None for v in row)
        ]
        if rows:
            parts.append(f"## Sheet: {ws.title}\n" + "\n".join(rows))
    wb.close()
    return "\n\n".join(parts)


@app.route("/chat/list_context", methods=["GET"])
def chat_list_context():
    chat_id = current_chat_id()
    files = chat_store.list_context(chat_id)
    stats = chat_store.stats(chat_id)
    stats.pop("files", None)  # the detailed list below supersedes the count
    return jsonify(status="success", files=files, count=len(files), **stats)


@app.route("/chat/delete_context", methods=["DELETE"])
def chat_delete_context():
    filename = request.args.get("filename")
    if not filename:
        return jsonify(status="error", message="Query parameter 'filename' is required"), 400
    result = chat_store.remove_context(current_chat_id(), filename)
    return jsonify(result), 200 if result["status"] == "success" else 404


@app.route("/chat/clear_context", methods=["POST"])
def chat_clear_context():
    return jsonify(chat_store.clear_context(current_chat_id()))


@app.route("/chat/new", methods=["POST"])
def chat_new():
    """Start a fresh conversation: drop history and any attached files."""
    old = session.get("chat_id")
    if old:
        chat_store.reset_chat(old)
    session["chat_id"] = uuid.uuid4().hex
    return jsonify(status="success", message="Started a new chat")


@app.route("/chat/status", methods=["GET"])
def chat_status():
    return jsonify(status="success", kb_ready=kb.ready, **chat_store.stats(current_chat_id()))


# ─── socket ───────────────────────────────────────────────────────────────────


def _run_off_hub(fn, *args):
    # Browser turns can run for a minute; blocking gevent's hub that long
    # misses Socket.IO pings and the client disconnects before the reply.
    if socketio.async_mode in ("gevent", "gevent_uwsgi"):
        import gevent

        return gevent.get_hub().threadpool.apply(fn, args)
    return fn(*args)


# Site-access prompts awaiting an answer: request id -> {sid, event, decision}.
_pending_permissions: Dict[str, Dict[str, Any]] = {}
_PERMISSION_DECISIONS = ("once", "always", "deny")


def _make_permission_asker(sid: str):
    """Build a blocking prompt that runs on the worker thread handling this turn."""
    hub = None
    if socketio.async_mode in ("gevent", "gevent_uwsgi"):
        import gevent

        hub = gevent.get_hub()  # captured on the hub thread, used from the worker

    def ask(domain: str, url: str) -> str:
        request_id = uuid.uuid4().hex
        entry = {"sid": sid, "event": threading.Event(), "decision": "deny"}
        _pending_permissions[request_id] = entry
        payload = {"id": request_id, "domain": domain, "url": url, "timeout": PERMISSION_TIMEOUT}

        def send():
            socketio.emit("browser_permission_request", payload, to=sid)

        if hub is not None:
            # Socket writes must happen on the hub thread, in a greenlet.
            import gevent

            hub.loop.run_callback_threadsafe(gevent.spawn, send)
        else:
            send()
        try:
            entry["event"].wait(PERMISSION_TIMEOUT)
            return entry["decision"]
        finally:
            _pending_permissions.pop(request_id, None)

    return ask


@socketio.on("browser_permission_response")
def handle_browser_permission_response(data):
    data = data or {}
    entry = _pending_permissions.get(str(data.get("id", "")))
    # Only the tab that was asked may answer.
    if entry is None or entry["sid"] != request.sid:
        return
    decision = data.get("decision")
    entry["decision"] = decision if decision in _PERMISSION_DECISIONS else "deny"
    entry["event"].set()


@socketio.on("voice_command")
def handle_voice_command(data):
    command = (data or {}).get("command", "")
    chat_id = current_chat_id()
    user_id = current_user_id()
    logger.info("[chat %s] received: %.80s", chat_id[:8], command)
    try:
        response = _run_off_hub(
            get_answer, command, chat_id, user_id, _make_permission_asker(request.sid)
        )
    except Exception:
        logger.exception("Unhandled error answering a message")
        response = "Something went wrong handling that message. Please try again."
    emit("response", {"result": response, "command": command})


# ─── settings: provider, model, credentials, output format ────────────────────


@app.route("/settings", methods=["GET"])
def get_settings():
    """Current configuration plus the provider catalogue. No raw keys."""
    return jsonify(status="success", **settings_store.public_state(current_user_id()))


@app.route("/settings", methods=["POST"])
def update_settings():
    """Patch settings. An `api_key` here is stored in memory only."""
    changes = request.get_json(silent=True) or {}
    if not isinstance(changes, dict):
        return jsonify(status="error", message="Expected a JSON object"), 400

    allowed = {
        "provider",
        "model",
        "base_url",
        "temperature",
        "max_tokens",
        "output_format",
        "api_key",
    }
    unknown = set(changes) - allowed
    if unknown:
        return jsonify(
            status="error", message=f"Unknown setting(s): {', '.join(sorted(unknown))}"
        ), 400

    result = settings_store.update(current_user_id(), changes)
    return jsonify(result), 200 if result["status"] == "success" else 400


@app.route("/settings/models", methods=["GET"])
def settings_models():
    """Ask the selected provider which models it actually serves."""
    user_id = current_user_id()
    settings = settings_store.get(user_id)
    provider_id = request.args.get("provider") or settings["provider"]
    try:
        models = providers.list_models(
            provider_id,
            api_key=settings_store.api_key(user_id, provider_id),
            base_url=request.args.get("base_url") or settings["base_url"],
        )
    except providers.ProviderError as e:
        return jsonify(status="error", message=str(e)), 400
    return jsonify(status="success", provider=provider_id, models=models)


@app.route("/settings/test", methods=["POST"])
def settings_test():
    """Round-trip a one-word prompt so a key can be verified before chatting."""
    user_id = current_user_id()
    settings = settings_store.get(user_id)
    body = request.get_json(silent=True) or {}
    provider_id = body.get("provider") or settings["provider"]
    result = providers.test_connection(
        provider_id,
        model=body.get("model") or (settings["model"] if provider_id == settings["provider"] else ""),
        api_key=settings_store.api_key(user_id, provider_id),
        base_url=body.get("base_url") or settings["base_url"],
    )
    return jsonify(result), 200 if result["status"] == "success" else 400


@app.route("/settings/key", methods=["DELETE"])
def settings_clear_key():
    provider_id = request.args.get("provider")
    if not provider_id:
        return jsonify(status="error", message="Query parameter 'provider' is required"), 400
    return jsonify(settings_store.clear_key(current_user_id(), provider_id))


@app.route("/settings/reset", methods=["POST"])
def settings_reset():
    """Forget settings and every stored key for this browser."""
    return jsonify(settings_store.reset(current_user_id()))


# ─── output: exports and image generation ─────────────────────────────────────


@app.route("/export/<fmt>", methods=["POST"])
def export(fmt):
    """Convert a table the browser scraped from an answer into CSV or XLSX."""
    payload = request.get_json(silent=True) or {}
    try:
        data, mimetype, filename = exporters.build(fmt.lower(), payload)
    except exporters.ExportError as e:
        return jsonify(status="error", message=str(e)), 400

    logger.info("Export %s: %s (%d bytes)", fmt, filename, len(data))
    return Response(
        data,
        mimetype=mimetype,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(data)),
            "Cache-Control": "no-store",
        },
    )


@app.route("/generate/image", methods=["POST"])
def generate_image():
    """Generate an image with the user's own provider.

    Only providers that support image generation can do this. The provider layer
    returns a clear message if the selected provider doesn't support it.
    """
    user_id = current_user_id()
    settings = settings_store.get(user_id)
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify(status="error", message="A prompt is required"), 400

    provider_id = settings["provider"]
    try:
        result = providers.generate_image(
            provider_id,
            prompt,
            api_key=settings_store.api_key(user_id, provider_id),
            base_url=settings["base_url"],
            model=body.get("model") or "gpt-image-1",
            size=body.get("size") or "1024x1024",
        )
    except providers.ProviderError as e:
        return jsonify(status="error", message=str(e)), 400
    return jsonify(status="success", **result)


if __name__ == "__main__":
    status = kb.get_context_status()
    if kb.ready:
        logger.info(
            "Organisation knowledge base ready: %d chunks from %d file(s) [%s]",
            status["total_chunks"],
            status["total_files"],
            status["embedding_model"],
        )
    elif kb.stale:
        logger.warning(
            "Organisation knowledge base index is STALE (built with a different "
            "embedding model). Retrieval is disabled until you run: "
            "python manage_kb.py reindex"
        )
    else:
        logger.warning("Organisation knowledge base unavailable: %s", kb.init_error)

    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    if debug:
        logger.warning("FLASK_DEBUG is on - do not use this in production")

    socketio.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 5000)),
        debug=debug,
        use_reloader=False,  # the reloader double-loads the KB singleton
        log_output=False,
    )
