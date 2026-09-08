#!/usr/bin/env python3
"""Salai - Flask + SocketIO chat over Azure OpenAI (via the Dentsu APIM gateway).

Two tiers of context, deliberately separate:

* **Organisation context** (``/admin/*``, requires ``ADMIN_API_KEY``) -- files
  admins curate for everyone. Persisted in Chroma, embedded, and retrieved by
  semantic similarity. Lives in ``knowledge_base.py``.

* **Chat context** (``/chat/*``, no admin key) -- files a user attaches to
  their own conversation. Held in memory, injected into the prompt verbatim,
  and discarded when the session ends. Lives in ``chat_store.py``.
"""
import functools
import hmac
import logging
import os
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import requests
from flask import Flask, jsonify, render_template, request, session
from flask_socketio import SocketIO, emit

# Load environment before importing modules that read it at import time.
from dotenv import load_dotenv

load_dotenv()

from chat_store import chat_store  # noqa: E402
from knowledge_base import SUPPORTED_EXTENSIONS, kb  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REQUIRED_VARS = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION",
    "SECRET_KEY",
    "ADMIN_API_KEY",
]

missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing)}. "
        "Copy .env.example to .env and fill it in."
    )

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 50 * 1024 * 1024))
REQUEST_TIMEOUT = int(os.getenv("LLM_TIMEOUT_SECONDS", 60))

SYSTEM_BASE = (
    "You are Salai, a helpful AI assistant for dentsu staff. "
    "Answer clearly and concisely, using Markdown where it helps.\n"
    # Retrieval runs per turn, so a follow-up question may carry no context
    # even though an earlier turn was grounded in real documents. Without this
    # the model disclaims its own previous answer ("I don't actually have
    # access to those documents"), which reads as a contradiction to the user.
    "Context is supplied per message, so documents quoted earlier in this "
    "conversation may not be repeated below. Trust your earlier answers and "
    "never claim you lack access to material you have already been shown."
)

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


# ─── prompt assembly ──────────────────────────────────────────────────────────


def build_messages(chat_id: str, user_input: str) -> Tuple[list, Dict[str, Any]]:
    """Compose the request payload for one turn.

    Context goes in the system message *only*. The user's message stays
    verbatim -- the previous implementation wrapped retrieved chunks into the
    user turn *and* repeated them in the system message, which doubled token
    cost and left the model unable to tell instructions from data.
    """
    meta: Dict[str, Any] = {"kb_status": "skipped", "kb_sources": [], "chat_files": 0}

    system_parts = [SYSTEM_BASE]

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
    messages.append({"role": "user", "content": user_input})
    return messages, meta


def get_answer(user_input: str, chat_id: str) -> str:
    """Run one conversation turn and persist it to the chat's history."""
    user_input = (user_input or "").strip()
    if not user_input:
        return "Please enter a question."

    messages, meta = build_messages(chat_id, user_input)
    logger.info(
        "[chat %s] kb=%s sources=%s attached=%d history=%d",
        chat_id[:8],
        meta["kb_status"],
        meta["kb_sources"] or "-",
        meta["chat_files"],
        len(messages) - 2,
    )

    endpoint = (
        f"{os.getenv('AZURE_OPENAI_ENDPOINT').rstrip('/')}/openai/deployments/"
        f"{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}/chat/completions"
        f"?api-version={os.getenv('AZURE_OPENAI_API_VERSION')}"
    )
    headers = {
        "x-brand": os.getenv("API_BRAND", "dentsu"),
        "x-service-line": os.getenv("API_SERVICE_LINE", "Functions"),
        "x-project": os.getenv("API_PROJECT", "test"),
        "Ocp-Apim-Subscription-Key": os.getenv("AZURE_OPENAI_API_KEY"),
        "api-version": os.getenv("API_GATEWAY_VERSION", "v15"),
        "Content-Type": "application/json",
    }
    payload = {
        "messages": messages,
        "temperature": float(os.getenv("LLM_TEMPERATURE", 0.7)),
        "max_completion_tokens": int(os.getenv("LLM_MAX_TOKENS", 2000)),
    }

    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.Timeout:
        logger.error("LLM request timed out after %ds", REQUEST_TIMEOUT)
        return "The request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        logger.error("LLM network error: %s", e)
        return "I could not reach the language model. Please check your connection and retry."

    if response.status_code != 200:
        # Never surface the raw gateway body: it can echo request headers.
        logger.error("LLM API error %s: %s", response.status_code, response.text[:500])
        return f"The language model returned an error (HTTP {response.status_code}). Please try again."

    try:
        answer = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as e:
        logger.error("Unexpected LLM response shape: %s", e)
        return "I received an unexpected response from the language model."

    # Record both sides of the turn. The old code stored only the assistant
    # message, which produced a malformed, assistant-only "conversation".
    chat_store.append_message(chat_id, "user", user_input)
    chat_store.append_message(chat_id, "assistant", answer)
    return answer


# ─── page ─────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    # Establish the chat id here, where setting a cookie actually works.
    if "chat_id" not in session:
        session["chat_id"] = uuid.uuid4().hex
        session.permanent = False
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
    file, error = _validated_upload()
    if error:
        return error

    tmp_path = _spool_to_temp(file)
    try:
        text = kb.extract_text(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if not text.strip():
        return jsonify(
            status="error",
            message="No extractable text found (scanned or image-only PDFs are not supported)",
        ), 400

    result = chat_store.add_context(chat_id, os.path.basename(file.filename), text)
    return jsonify(result), 200 if result["status"] == "success" else 400


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


@socketio.on("voice_command")
def handle_voice_command(data):
    command = (data or {}).get("command", "")
    chat_id = current_chat_id()
    logger.info("[chat %s] received: %.80s", chat_id[:8], command)
    try:
        response = get_answer(command, chat_id)
    except Exception:
        logger.exception("Unhandled error answering a message")
        response = "Something went wrong handling that message. Please try again."
    emit("response", {"result": response, "command": command})


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
