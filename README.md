# Salai

A Flask + SocketIO chat assistant over Azure OpenAI (via the dentsu APIM gateway),
with two independent tiers of document context.

## The two context tiers

These are deliberately separate, with different scope, storage and permissions.

| | **Organisation context** | **Chat context** |
|---|---|---|
| Who can add | admins only (`ADMIN_API_KEY`) | anyone using the app |
| Endpoints | `/admin/*` | `/chat/*` |
| Visible to | everyone, in every conversation | only the chat it was attached to |
| Storage | ChromaDB on disk (`vectorstore/`) plus the original file in `uploads/` | process memory only |
| How it reaches the model | embedded, then retrieved by semantic similarity per question | injected into the prompt verbatim |
| Lifetime | until an admin deletes it | until "New chat", or 8h idle |

**Organisation context** is the curated, shared knowledge base — brand guidelines,
policies, standard briefs. Index it once and every user's questions are searched
against it.

**Chat context** is "here is my document, answer questions about it". It never
touches the shared index, is not visible to anyone else, and is discarded when
the conversation ends. Because it is injected whole rather than retrieved, the
model always sees the entire file — no retrieval misses on something you just
attached.

## Setup

### 1. Install

```bash
pip install -r requirements.txt
```

Embeddings run locally via ChromaDB's bundled ONNX build of `all-MiniLM-L6-v2`.
The first run downloads it (~80 MB) to `~/.cache/chroma/` and works offline
afterwards. No PyTorch required.

### 2. Configure

```bash
cp .env.example .env
```

Fill in the required values:

| Variable | Purpose |
|---|---|
| `AZURE_OPENAI_API_KEY` | APIM subscription key |
| `AZURE_OPENAI_ENDPOINT` | e.g. `https://ai-api-dev.dentsu.com` |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | e.g. `GPT4o128k` |
| `AZURE_OPENAI_API_VERSION` | e.g. `2024-10-21` |
| `SECRET_KEY` | Flask session signing key |
| `ADMIN_API_KEY` | gates every `/admin` endpoint |

Generate the two secrets with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

`.env.example` documents the optional retrieval, limit and server settings.

### 3. Run

```bash
python app.py
```

Then open <http://localhost:5000>.

### 4. Add organisation context

Either through the UI — **📎 Context → Organisation context**, enter the admin
key, drag files in — or from the command line:

```bash
python manage_kb.py index --folder ./docs
python manage_kb.py status
python manage_kb.py query "what are the email letterhead rules?"
```

## Managing the knowledge base

```bash
python manage_kb.py status              # what is indexed
python manage_kb.py index --folder DIR  # index every supported file in DIR
python manage_kb.py reindex             # wipe and rebuild from uploads/
python manage_kb.py reset               # drop every vector
python manage_kb.py prune               # delete uploads/ files no longer indexed
python manage_kb.py query "question"    # test retrieval without the LLM
```

`query` is the quickest way to sanity-check relevance: it prints the retrieval
status and matched sources without calling the language model.

## Supported file types

`.txt` `.md` `.csv` `.pdf` `.docx`

PDFs must contain a text layer. Scanned or image-only PDFs extract nothing —
there is no OCR.

## API

All `/admin` endpoints require the key in an **`X-API-Key` header**. It is not
accepted as a query parameter, because URLs end up in server access logs,
browser history and `Referer` headers.

### Organisation context (admin)

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/admin/kb_status` | index health |
| `GET` | `/admin/context_status` | file/chunk counts, storage size |
| `GET` | `/admin/list_context` | indexed documents |
| `POST` | `/admin/upload_context` | upload and index a file → `202` + `job_id` |
| `GET` | `/admin/upload_status/<job_id>` | poll an indexing job |
| `DELETE` | `/admin/delete_context?source=NAME` | remove a document |
| `POST` | `/admin/index_kb?folder=PATH` | bulk-index a folder |
| `POST` | `/admin/prune_uploads` | delete unreferenced upload files |

Indexing is asynchronous. `POST /admin/upload_context` returns `202` with a
`job_id`; poll `/admin/upload_status/<job_id>` until `status` is no longer
`processing`.

```bash
curl -X POST http://localhost:5000/admin/upload_context \
  -H "X-API-Key: $ADMIN_API_KEY" -F "file=@guidelines.pdf"
```

### Chat context (no admin key)

Scoped to the caller's session cookie.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat/upload_context` | attach a file to this chat |
| `GET` | `/chat/list_context` | files attached to this chat |
| `DELETE` | `/chat/delete_context?filename=NAME` | detach one file |
| `POST` | `/chat/clear_context` | detach all files |
| `POST` | `/chat/new` | new chat: clears history and files |
| `GET` | `/chat/status` | message/file counts |

## Tests

```bash
python -m pytest test_smoke.py -v
```

46 tests covering extraction, chunking, retrieval relevance, per-chat
isolation, history persistence, prompt assembly and endpoint auth. No network
calls — the suite uses a scratch vector store and never touches the live index.

## How retrieval works

1. The question is embedded locally with `all-MiniLM-L6-v2` (384 dimensions).
2. ChromaDB returns the `KB_RETRIEVAL_K` nearest chunks by cosine distance.
3. Chunks further than `KB_MAX_DISTANCE` are **discarded**. If nothing survives,
   no context is added and the model answers from general knowledge.
4. Surviving chunks go into the system message with their source filenames.

Step 3 matters. Without a distance threshold every question retrieves
something, so "hello" would pull in unrelated policy text and the model would
be instructed to answer from it. On the current corpus an on-topic query scores
~0.44 and an off-topic one ~0.93; the default cut-off of `0.75` separates them
with margin. Tune it with `KB_MAX_DISTANCE` — lower is stricter.

### Why embeddings are local

The APIM gateway rejects every embedding deployment:

```
403 {"error":{"message":"Model not allowed. You can only use o1, o3,
     o3-deep-research, or GPT-4 models.","code":"model_not_allowed"}}
```

So there is no remote embedding option, and embeddings run on the local ONNX
model instead. If an embedding deployment is ever allowed through the gateway,
swap the embedding function in `knowledge_base.py`, bump `EMBEDDING_ID`, and
run `python manage_kb.py reindex`.

`EMBEDDING_ID` is recorded in `vectorstore/.embedding_model`. If it does not
match the running code, retrieval is **disabled** rather than served from a
mismatched vector space, and the app tells you to reindex.

## Architecture

| File | Responsibility |
|---|---|
| `app.py` | Flask routes, SocketIO handler, auth, prompt assembly |
| `knowledge_base.py` | organisation index: extraction, chunking, embedding, retrieval |
| `chat_store.py` | per-chat history and attached files, with TTL and size caps |
| `manage_kb.py` | knowledge base CLI |
| `templates/index.html` | the entire UI |
| `test_smoke.py` | test suite |

## Deployment notes

- `chat_store` is **in-process**. That suits the single-process `socketio.run`
  server used here; running multiple workers needs a shared backend such as
  Redis, or users will lose history and attachments between requests.
- Set `FLASK_DEBUG=false` (the default) outside local development.
- Set `SESSION_COOKIE_SECURE=true` when serving over HTTPS.
- There is **no user authentication** on the chat itself. Anyone who can reach
  the port can chat and read the organisation context. Put it behind SSO or a
  network boundary before exposing it beyond localhost.

## Troubleshooting

**"Index is STALE"** — built with a different embedding model. Run
`python manage_kb.py reindex`.

**Retrieval returns nothing for a question you know is covered** — check with
`python manage_kb.py query "your question"`. If the status is `empty`, the
match was outside the distance threshold; raise `KB_MAX_DISTANCE` slightly, or
confirm the document actually has a text layer via `manage_kb.py status`.

**"No extractable text found"** — the PDF is scanned/image-only. There is no
OCR; supply a text-based version.

**Upload rejected with 413** — larger than `MAX_UPLOAD_BYTES` (50 MB default).

**Admin endpoints return 401** — the key must be in the `X-API-Key` header;
`?api_key=` is deliberately not accepted.

## Not implemented

Earlier versions of this README documented SharePoint and Microsoft Graph
integration — indexing directly from a SharePoint folder and storing the vector
store in an `AgentKai_VectorStore` folder. None of that exists in the code.
Documents are uploaded through the UI or CLI and stored locally. The
`AZURE_CLIENT_ID` / `AZURE_CLIENT_SECRET` / `AZURE_TENANT_ID` /
`SHAREPOINT_SITE_URL` variables are unused and commented out in
`.env.example`.
