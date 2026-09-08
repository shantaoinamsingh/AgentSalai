#!/usr/bin/env python3
"""Smoke tests: text extraction, chunking, retrieval gating, per-chat context, auth.

Run with:  python -m pytest test_smoke.py -v

No network calls -- embeddings are local and the LLM call is stubbed, so this
suite is safe to run in CI.
"""
import os
import sys
import tempfile

import pytest

# Redirect storage to a scratch directory *before* importing the app, so the
# suite can never reset or delete the real index or uploaded files.
_SCRATCH = tempfile.mkdtemp(prefix="salai-test-")
os.environ["VECTORSTORE_DIR"] = os.path.join(_SCRATCH, "vectorstore")
os.environ["UPLOADS_DIR"] = os.path.join(_SCRATCH, "uploads")

os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-key")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://example.invalid")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT_NAME", "test-deployment")
os.environ.setdefault("AZURE_OPENAI_API_VERSION", "2024-10-21")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat_store import MAX_CONTEXT_FILES, ChatStore  # noqa: E402
from knowledge_base import SimpleTextSplitter, kb  # noqa: E402


# ─── chunking ─────────────────────────────────────────────────────────────────


def test_short_text_is_one_chunk():
    assert SimpleTextSplitter().split_text("hello world") == ["hello world"]


def test_empty_text_yields_no_chunks():
    assert SimpleTextSplitter().split_text("") == []
    assert SimpleTextSplitter().split_text("   \n  ") == []


def test_long_text_splits_and_covers_content():
    text = ". ".join(f"sentence number {i} about media planning" for i in range(400))
    chunks = SimpleTextSplitter(chunk_size=500, chunk_overlap=100).split_text(text)
    assert len(chunks) > 1
    assert "sentence number 0" in chunks[0]
    assert "sentence number 399" in chunks[-1]


def test_splitter_terminates_on_pathological_input():
    """A boundary inside the overlap window must not stall the loop."""
    text = "a" * 5000
    chunks = SimpleTextSplitter(chunk_size=100, chunk_overlap=90).split_text(text)
    assert len(chunks) < 500  # would hang or explode if start never advanced


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        SimpleTextSplitter(chunk_size=100, chunk_overlap=100)


# ─── extraction ───────────────────────────────────────────────────────────────


def test_extract_txt(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("Media plans are due on Friday.", encoding="utf-8")
    assert "due on Friday" in kb.extract_text(str(p))


def test_extract_markdown_strips_syntax(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nSome **bold** and a [link](http://x.com).", encoding="utf-8")
    out = kb.extract_text(str(p))
    assert "Title" in out and "bold" in out and "link" in out
    assert "#" not in out and "**" not in out and "http://x.com" not in out


def test_extract_csv_handles_quoted_commas(tmp_path):
    """The old hand-rolled split(',') mangled any quoted field."""
    p = tmp_path / "data.csv"
    p.write_text('client,budget\n"Acme, Inc.",50000\n', encoding="utf-8")
    out = kb.extract_text(str(p))
    assert "client: Acme, Inc." in out
    assert "budget: 50000" in out


def test_extract_unsupported_returns_empty(tmp_path):
    p = tmp_path / "image.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert kb.extract_text(str(p)) == ""


# ─── retrieval ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def indexed_kb():
    """Index two clearly distinct documents into a clean store."""
    if kb.collection is None:
        pytest.skip(f"knowledge base unavailable: {kb.init_error}")

    kb.reset_index()
    kb._add_chunks(
        "email_guidelines.txt",
        [
            "All dentsu email signatures must use the approved logo header. "
            "Letterhead artwork is available as JPGs in the dentsu brand hub."
        ],
        {"uploaded_at": "test"},
    )
    kb._add_chunks(
        "expenses_policy.txt",
        ["Travel expense claims must be submitted within 30 days with receipts attached."],
        {"uploaded_at": "test"},
    )
    yield kb
    kb.reset_index()


def test_relevant_query_retrieves_right_source(indexed_kb):
    result = indexed_kb.query("what are the rules for email letterhead and logos?")
    assert result["status"] == "ok"
    assert "email_guidelines.txt" in result["sources"]


def test_relevant_query_finds_other_document(indexed_kb):
    result = indexed_kb.query("how long do I have to claim travel expenses?")
    assert result["status"] == "ok"
    assert "expenses_policy.txt" in result["sources"]


def test_greeting_retrieves_nothing(indexed_kb):
    """The core regression: 'hello' used to return arbitrary chunks.

    With hallucinated embeddings every query matched, so the model was handed
    irrelevant context and told to answer from it.
    """
    result = indexed_kb.query("hello")
    assert result["status"] == "empty"
    assert result["context"] == ""


def test_unrelated_query_retrieves_nothing(indexed_kb):
    result = indexed_kb.query("what is the best recipe for sourdough bread?")
    assert result["status"] == "empty"


def test_empty_query_is_safe(indexed_kb):
    assert indexed_kb.query("")["status"] == "empty"


# ─── ingest / delete round trip ───────────────────────────────────────────────


def test_ingest_then_remove_cleans_up_file(tmp_path, indexed_kb):
    """Deleting a document must also delete its stored upload.

    The old matcher compared `stored.endswith("_" + source)`, which never
    matched because stored names are sanitised while the source keeps spaces,
    so every upload leaked its file on disk.
    """
    src = tmp_path / "quarterly report notes.txt"
    src.write_text("Quarterly revenue rose in the EMEA region.", encoding="utf-8")
    key = os.environ["ADMIN_API_KEY"]

    result = indexed_kb.ingest_uploaded_file(str(src), "quarterly report notes.txt", key)
    assert result["status"] == "success", result
    assert result["chunks"] >= 1

    from knowledge_base import LOCAL_UPLOADS_DIR

    manifest = indexed_kb._load_manifest()
    assert "quarterly report notes.txt" in manifest
    stored_path = os.path.join(
        LOCAL_UPLOADS_DIR, manifest["quarterly report notes.txt"]["stored_path"]
    )
    assert os.path.isfile(stored_path)

    removal = indexed_kb.remove_document("quarterly report notes.txt")
    assert removal["status"] == "success"
    assert removal["file_removed"] is True
    assert not os.path.exists(stored_path)
    assert "quarterly report notes.txt" not in indexed_kb._load_manifest()


def test_ingest_rejects_unsupported_type(tmp_path, indexed_kb):
    p = tmp_path / "thing.exe"
    p.write_bytes(b"MZ")
    result = indexed_kb.ingest_uploaded_file(str(p), "thing.exe", os.environ["ADMIN_API_KEY"])
    assert result["status"] == "error"
    assert "Unsupported" in result["message"]


# ─── admin auth ───────────────────────────────────────────────────────────────


def test_wrong_admin_key_is_rejected(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("data", encoding="utf-8")
    with pytest.raises(PermissionError):
        kb.ingest_uploaded_file(str(p), "x.txt", "definitely-not-the-key")


def test_empty_admin_key_is_rejected():
    assert kb.verify_admin_access("") is False
    assert kb.verify_admin_access(None) is False


def test_correct_admin_key_is_accepted():
    assert kb.verify_admin_access(os.environ["ADMIN_API_KEY"]) is True


# ─── per-chat context ─────────────────────────────────────────────────────────


def test_chat_context_is_isolated_between_chats():
    store = ChatStore()
    store.add_context("chat-a", "a.txt", "alpha content")
    store.add_context("chat-b", "b.txt", "beta content")

    assert "alpha content" in store.build_context_text("chat-a")
    assert "beta content" not in store.build_context_text("chat-a")
    assert "beta content" in store.build_context_text("chat-b")


def test_chat_context_reupload_replaces():
    store = ChatStore()
    store.add_context("c", "notes.txt", "first version")
    store.add_context("c", "notes.txt", "second version")
    text = store.build_context_text("c")
    assert "second version" in text
    assert "first version" not in text
    assert len(store.list_context("c")) == 1


def test_chat_context_remove_and_clear():
    store = ChatStore()
    store.add_context("c", "one.txt", "content one")
    store.add_context("c", "two.txt", "content two")

    assert store.remove_context("c", "one.txt")["status"] == "success"
    assert store.remove_context("c", "one.txt")["status"] == "error"  # already gone
    assert len(store.list_context("c")) == 1

    assert store.clear_context("c")["removed"] == 1
    assert store.build_context_text("c") == ""


def test_chat_context_rejects_empty_text():
    store = ChatStore()
    assert store.add_context("c", "blank.txt", "   ")["status"] == "error"


def test_chat_context_enforces_file_cap():
    store = ChatStore()
    for i in range(MAX_CONTEXT_FILES):
        assert store.add_context("c", f"f{i}.txt", f"content {i}")["status"] == "success"
    overflow = store.add_context("c", "one-too-many.txt", "content")
    assert overflow["status"] == "error"


def test_chat_context_truncates_oversized_file():
    store = ChatStore()
    result = store.add_context("c", "big.txt", "x" * 500_000)
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["chars"] < 500_000


def test_reset_chat_clears_history_and_files():
    store = ChatStore()
    store.append_message("c", "user", "hi")
    store.add_context("c", "f.txt", "content")
    store.reset_chat("c")
    assert store.get_history("c") == []
    assert store.build_context_text("c") == ""


# ─── chat history ─────────────────────────────────────────────────────────────


def test_history_records_both_roles_in_order():
    """Only the assistant turn used to be stored, yielding a malformed history."""
    store = ChatStore()
    store.append_message("c", "user", "what is our email policy?")
    store.append_message("c", "assistant", "Use the approved logo header.")
    history = store.get_history("c")
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "what is our email policy?"


def test_history_is_trimmed_to_the_cap():
    store = ChatStore()
    for i in range(100):
        store.append_message("c", "user", f"message {i}")
    history = store.get_history("c")
    assert len(history) <= 20
    assert history[-1]["content"] == "message 99"  # newest kept


def test_history_survives_repeated_reads():
    """Regression for the cookie-session bug: writes must actually persist."""
    store = ChatStore()
    store.append_message("c", "user", "remember this")
    store.get_history("c")
    assert store.get_history("c")[0]["content"] == "remember this"


# ─── prompt assembly ──────────────────────────────────────────────────────────


def test_prompt_keeps_user_message_verbatim(indexed_kb, monkeypatch):
    """Context belongs in the system message; the user turn must stay clean.

    The old code inlined retrieved chunks into the user's own message *and*
    repeated them in the system prompt.
    """
    import app as app_module

    question = "what are the rules for email letterhead and logos?"
    with app_module.app.test_request_context("/"):
        messages, meta = app_module.build_messages("chat-x", question)

    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": question}
    assert meta["kb_status"] == "ok"
    assert "logo header" in messages[0]["content"]
    # Context must appear exactly once, not duplicated across roles.
    assert sum("brand hub" in m["content"] for m in messages) == 1


def test_prompt_omits_context_when_nothing_relevant(indexed_kb):
    import app as app_module

    with app_module.app.test_request_context("/"):
        messages, meta = app_module.build_messages("chat-y", "hello there")

    assert meta["kb_status"] == "empty"
    assert meta["chat_files"] == 0
    assert len(messages) == 2  # system + user only
    assert "brand hub" not in messages[0]["content"]


def test_prompt_includes_per_chat_files(indexed_kb):
    import app as app_module
    from chat_store import chat_store

    chat_store.reset_chat("chat-z")
    chat_store.add_context("chat-z", "my_brief.txt", "The client is launching in Q3.")
    try:
        with app_module.app.test_request_context("/"):
            messages, meta = app_module.build_messages("chat-z", "when is the launch?")
        assert meta["chat_files"] == 1
        assert "my_brief.txt" in messages[0]["content"]
        assert "launching in Q3" in messages[0]["content"]
    finally:
        chat_store.reset_chat("chat-z")


# ─── HTTP surface ─────────────────────────────────────────────────────────────


@pytest.fixture
def client():
    import app as app_module

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/admin/list_context"),
        ("get", "/admin/context_status"),
        ("get", "/admin/kb_status"),
        ("post", "/admin/upload_context"),
        ("delete", "/admin/delete_context?source=x"),
    ],
)
def test_admin_endpoints_require_a_key(client, method, path):
    response = getattr(client, method)(path)
    assert response.status_code == 401


def test_admin_endpoint_rejects_bad_key(client):
    response = client.get("/admin/list_context", headers={"X-API-Key": "wrong"})
    assert response.status_code == 403


def test_admin_key_in_query_string_is_not_accepted(client):
    """Keys in URLs leak into logs and history, so only the header is honoured."""
    key = os.environ["ADMIN_API_KEY"]
    assert client.get(f"/admin/list_context?api_key={key}").status_code == 401


def test_admin_endpoint_accepts_header_key(client):
    response = client.get("/admin/list_context", headers={"X-API-Key": os.environ["ADMIN_API_KEY"]})
    assert response.status_code == 200
    assert response.get_json()["status"] == "success"


def test_chat_endpoints_need_no_admin_key(client):
    """Per-chat context is scoped to the session, so it must not require a key."""
    response = client.get("/chat/list_context")
    assert response.status_code == 200
    assert response.get_json()["status"] == "success"


def test_chat_upload_rejects_unsupported_type(client):
    import io

    response = client.post(
        "/chat/upload_context",
        data={"file": (io.BytesIO(b"MZ"), "malware.exe")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Unsupported" in response.get_json()["message"]


def test_chat_upload_requires_a_file(client):
    response = client.post("/chat/upload_context", data={}, content_type="multipart/form-data")
    assert response.status_code == 400


def test_chat_upload_and_listing_round_trip(client):
    import io

    upload = client.post(
        "/chat/upload_context",
        data={"file": (io.BytesIO(b"The campaign launches in Q3 2026."), "brief.txt")},
        content_type="multipart/form-data",
    )
    assert upload.status_code == 200, upload.get_json()
    assert upload.get_json()["status"] == "success"

    listing = client.get("/chat/list_context").get_json()
    assert [f["filename"] for f in listing["files"]] == ["brief.txt"]

    removal = client.delete("/chat/delete_context?filename=brief.txt")
    assert removal.status_code == 200
    assert client.get("/chat/list_context").get_json()["count"] == 0


def test_new_chat_clears_attached_files(client):
    import io

    client.post(
        "/chat/upload_context",
        data={"file": (io.BytesIO(b"some notes for this chat"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert client.get("/chat/list_context").get_json()["count"] == 1

    assert client.post("/chat/new").status_code == 200
    assert client.get("/chat/list_context").get_json()["count"] == 0


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Salai" in response.data
