#!/usr/bin/env python3
"""Organisation-level knowledge base for Salai.

This module owns the *persistent, org-wide* context that admins curate.
Per-chat file attachments are a separate concern and live in chat_store.py.

Retrieval design notes
----------------------
Embeddings run **locally** via the ONNX build of ``all-MiniLM-L6-v2`` that
ships with chromadb. The Dentsu APIM gateway rejects every embedding
deployment ("Model not allowed. You can only use o1, o3, o3-deep-research,
or GPT-4 models"), so a remote embedding call is not an option. The previous
implementation worked around this by asking GPT-4o to emit "1536 floats" and
falling back to an all-zero vector on failure, which made similarity search
meaningless -- any query returned arbitrary chunks.

Because the local model is 384-dimensional, an index built by the old code is
incompatible *and* semantically worthless. We record the embedding identity in
a marker file and refuse to serve retrieval from a stale index rather than
silently returning garbage. Run ``python manage_kb.py reindex`` to rebuild.
"""
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Overridable so the test suite can point at a scratch directory instead of
# operating on the live index.
if os.getenv("VERCEL"):
    STORAGE_BASE_DIR = "/tmp/agent_salai"
else:
    STORAGE_BASE_DIR = BASE_DIR
    
LOCAL_UPLOADS_DIR = os.getenv("UPLOADS_DIR") or os.path.join(BASE_DIR, "uploads")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR") or os.path.join(BASE_DIR, "vectorstore")
MANIFEST_PATH = os.path.join(LOCAL_UPLOADS_DIR, "_manifest.json")
EMBEDDING_MARKER_PATH = os.path.join(VECTORSTORE_DIR, ".embedding_model")

COLLECTION_NAME = "agent_kb"

# Identity of the embedding space. Bump this if the model or its dimensions
# change so stale indexes are detected instead of silently mixed.
EMBEDDING_ID = "onnx-all-MiniLM-L6-v2-384"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".pdf", ".docx", ".doc"}

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 50 * 1024 * 1024))

CHUNK_SIZE = int(os.getenv("KB_CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("KB_CHUNK_OVERLAP", 200))

# Number of chunks to retrieve, and the maximum cosine distance at which a
# chunk is still considered relevant. Measured on this corpus: an on-topic
# query scores ~0.44 and an off-topic one ~0.93, so 0.75 separates them with
# margin on both sides. Without this gate every query -- including "hello" --
# returned context and the model was told to answer from it.
RETRIEVAL_K = int(os.getenv("KB_RETRIEVAL_K", 4))
MAX_DISTANCE = float(os.getenv("KB_MAX_DISTANCE", 0.75))

os.makedirs(LOCAL_UPLOADS_DIR, exist_ok=True)


class SimpleTextSplitter:
    """Chunk text on sentence/paragraph boundaries where possible."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                boundary = max(
                    text.rfind(". ", start, end),
                    text.rfind(".\n", start, end),
                    text.rfind("\n\n", start, end),
                    text.rfind("\n", start, end),
                )
                if boundary > start + self.chunk_size // 2:
                    end = boundary + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            # Always advance, even if a boundary landed inside the overlap
            # window, so we cannot loop forever on pathological input.
            start = max(end - self.chunk_overlap, start + 1)
        return chunks


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class KnowledgeBase:
    """Persistent, admin-curated organisation context."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.client = None
        self.collection = None
        self.stale = False
        self.init_error: Optional[str] = None
        self._init()

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def admin_api_key(self) -> Optional[str]:
        # Read lazily: app.py loads .env before importing this module, but
        # tests and scripts may set it afterwards.
        return os.getenv("ADMIN_API_KEY")

    @property
    def ready(self) -> bool:
        return self.collection is not None and not self.stale

    def _init(self) -> None:
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            os.makedirs(VECTORSTORE_DIR, exist_ok=True)
            self._check_embedding_marker()

            self.embedding_function = embedding_functions.ONNXMiniLM_L6_V2()
            self.client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Knowledge base ready (embedding=%s, chunks=%d, stale=%s)",
                EMBEDDING_ID,
                self.collection.count(),
                self.stale,
            )
        except Exception as e:
            self.init_error = str(e)
            logger.error("Knowledge base init failed: %s", e)

    def _check_embedding_marker(self) -> None:
        """Detect an index built with a different (or the broken legacy) model."""
        existing = None
        if os.path.exists(EMBEDDING_MARKER_PATH):
            try:
                with open(EMBEDDING_MARKER_PATH, "r", encoding="utf-8") as f:
                    existing = f.read().strip()
            except Exception:
                existing = None

        legacy_db = os.path.exists(os.path.join(VECTORSTORE_DIR, "chroma.sqlite3"))

        if existing == EMBEDDING_ID:
            return

        if existing is None and not legacy_db:
            self._write_embedding_marker()
            return

        self.stale = True
        logger.error(
            "Vector index was built with embedding %r but this build uses %r. "
            "Retrieval is DISABLED to avoid returning meaningless matches. "
            "Rebuild with:  python manage_kb.py reindex",
            existing or "legacy-llm-generated-floats",
            EMBEDDING_ID,
        )

    def _write_embedding_marker(self) -> None:
        try:
            os.makedirs(VECTORSTORE_DIR, exist_ok=True)
            with open(EMBEDDING_MARKER_PATH, "w", encoding="utf-8") as f:
                f.write(EMBEDDING_ID)
        except Exception as e:
            logger.warning("Could not write embedding marker: %s", e)

    def reset_index(self) -> None:
        """Drop every vector and re-create an empty collection."""
        with self._lock:
            if self.client is None:
                raise RuntimeError(f"Knowledge base not initialised: {self.init_error}")
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass  # collection may not exist yet
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
            self.stale = False
            self._write_embedding_marker()
            logger.info("Vector index reset")

    # ── auth ─────────────────────────────────────────────────────────────────

    def verify_admin_access(self, api_key: Optional[str]) -> bool:
        import hmac

        configured = self.admin_api_key
        if not configured or not api_key:
            return False
        return hmac.compare_digest(str(api_key), str(configured))

    def _require_admin(self, api_key: Optional[str]) -> None:
        if not self.verify_admin_access(api_key):
            raise PermissionError("Invalid admin API key")

    # ── manifest ─────────────────────────────────────────────────────────────
    # Maps display name -> stored file. Needed for two reasons: deleting an
    # upload used to try `filename.endswith("_" + source)`, which never matched
    # because stored names are sanitised (spaces -> underscores) while the
    # source keeps its spaces -- so every upload leaked its file. It also lets
    # `reindex` rebuild from disk without re-uploading.

    def _load_manifest(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(MANIFEST_PATH):
            return {}
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning("Could not read upload manifest: %s", e)
            return {}

    def _save_manifest(self, manifest: Dict[str, Dict[str, Any]]) -> None:
        try:
            tmp = MANIFEST_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            os.replace(tmp, MANIFEST_PATH)
        except Exception as e:
            logger.warning("Could not write upload manifest: %s", e)

    # ── extraction ───────────────────────────────────────────────────────────

    def extract_text(self, path: str) -> str:
        """Extract plain text from a supported document. Returns "" on failure."""
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".txt",):
                return self._read_text(path).strip()

            if ext in (".md", ".markdown"):
                text = self._read_text(path)
                text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
                text = re.sub(r"#{1,6}\s+", "", text)
                text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
                text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
                return text.strip()

            if ext == ".csv":
                return self._extract_csv(path)

            if ext == ".pdf":
                return self._extract_pdf(path)

            if ext in (".docx", ".doc"):
                return self._extract_docx(path)

            logger.warning("Unsupported extension for extraction: %s", ext)
        except Exception as e:
            logger.warning("Error extracting text from %s: %s", path, e)
        return ""

    @staticmethod
    def _read_text(path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    @staticmethod
    def _extract_csv(path: str) -> str:
        """Render rows as "column: value" lines, which embed better than raw CSV."""
        import csv

        out: List[str] = []
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)  # handles quoted fields containing commas
            for row in reader:
                cells = [
                    f"{(k or '').strip()}: {(v or '').strip()}"
                    for k, v in row.items()
                    if v and str(v).strip()
                ]
                if cells:
                    out.append("\n".join(cells))
        return "\n\n".join(out).strip()

    @staticmethod
    def _extract_pdf(path: str) -> str:
        try:
            import PyPDF2
        except ImportError:
            logger.warning("PyPDF2 not installed; cannot read PDF %s", path)
            return ""
        parts: List[str] = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                except Exception as e:
                    logger.warning("PDF page %d of %s unreadable: %s", i, path, e)
                    continue
                if page_text and page_text.strip():
                    parts.append(page_text.strip())
        return "\n\n".join(parts).strip()

    @staticmethod
    def _extract_docx(path: str) -> str:
        try:
            from docx import Document
        except ImportError:
            logger.warning("python-docx not installed; cannot read %s", path)
            return ""
        parts: List[str] = []
        doc = Document(path)
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text.strip())
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts).strip()

    # ── ingestion ────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        base = os.path.basename(name or "upload")
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "upload"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{stamp}_{uuid.uuid4().hex[:8]}_{base}"

    @staticmethod
    def _file_digest(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    def _add_chunks(self, source: str, chunks: List[str], extra_meta: Dict[str, Any]) -> None:
        """Embed and store chunks under a stable, source-derived id scheme."""
        ids = [f"{source}::{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "chunk": i, **extra_meta} for i in range(len(chunks))]
        # Batch so a large document does not build one huge ONNX input.
        batch = 128
        for i in range(0, len(chunks), batch):
            self.collection.add(
                ids=ids[i : i + batch],
                documents=chunks[i : i + batch],
                metadatas=metadatas[i : i + batch],
            )

    def ingest_uploaded_file(
        self, file_path: str, original_filename: str, api_key: Optional[str]
    ) -> Dict[str, Any]:
        """Store, extract, chunk and index one uploaded document (admin only)."""
        self._require_admin(api_key)

        if self.collection is None:
            return {"status": "error", "message": f"Knowledge base not initialised: {self.init_error}"}
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}

        ext = os.path.splitext(original_filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return {
                "status": "error",
                "message": f"Unsupported file type: {ext or '(none)'}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            }

        size = os.path.getsize(file_path)
        if size > MAX_UPLOAD_BYTES:
            return {
                "status": "error",
                "message": f"File is {size / 1e6:.1f} MB, over the {MAX_UPLOAD_BYTES / 1e6:.0f} MB limit",
            }

        display_name = os.path.basename(original_filename)

        with self._lock:
            # Replace any prior copy of this document, including its file on
            # disk, so re-uploading does not accumulate duplicates.
            self._remove_document_unlocked(display_name)

            safe_name = self._sanitize_filename(display_name)
            stored_path = os.path.join(LOCAL_UPLOADS_DIR, safe_name)
            try:
                shutil.copy2(file_path, stored_path)
            except Exception as e:
                logger.error("Could not store upload %s: %s", display_name, e)
                return {"status": "error", "message": f"Could not store file: {e}"}

            content = self.extract_text(stored_path)
            if not content.strip():
                try:
                    os.remove(stored_path)
                except OSError:
                    pass
                return {
                    "status": "error",
                    "message": "No extractable text found (scanned/image-only PDFs are not supported)",
                }

            chunks = SimpleTextSplitter().split_text(content)
            if not chunks:
                try:
                    os.remove(stored_path)
                except OSError:
                    pass
                return {"status": "error", "message": "Document produced no indexable chunks"}

            uploaded_at = _utcnow()
            try:
                self._add_chunks(
                    display_name,
                    chunks,
                    {"stored_path": safe_name, "uploaded_at": uploaded_at},
                )
            except Exception as e:
                logger.error("Error indexing %s: %s", display_name, e)
                try:
                    os.remove(stored_path)
                except OSError:
                    pass
                return {"status": "error", "message": f"Indexing failed: {e}"}

            manifest = self._load_manifest()
            manifest[display_name] = {
                "stored_path": safe_name,
                "chunks": len(chunks),
                "chars": len(content),
                "bytes": size,
                "sha256": self._file_digest(stored_path),
                "uploaded_at": uploaded_at,
            }
            self._save_manifest(manifest)

            # A successful write means the on-disk index now matches this model.
            self.stale = False
            self._write_embedding_marker()

        logger.info("Indexed %d chunks from %s", len(chunks), display_name)
        return {
            "status": "success",
            "filename": display_name,
            "chunks": len(chunks),
            "message": f"Indexed {len(chunks)} chunks from {display_name}",
        }

    def index_documents(self, folder_path: str, api_key: Optional[str]) -> Dict[str, Any]:
        """Bulk-index every supported document under a local folder (admin only)."""
        self._require_admin(api_key)

        if self.collection is None:
            return {"status": "error", "message": f"Knowledge base not initialised: {self.init_error}"}
        if not folder_path or not os.path.isdir(folder_path):
            return {"status": "error", "message": f"Not a directory: {folder_path!r}"}

        indexed, skipped = [], []
        for root, _dirs, files in os.walk(folder_path):
            for name in sorted(files):
                if name.startswith("_") or name.startswith("."):
                    continue
                path = os.path.join(root, name)
                if os.path.splitext(name)[1].lower() not in SUPPORTED_EXTENSIONS:
                    skipped.append({"file": name, "reason": "unsupported type"})
                    continue
                result = self.ingest_uploaded_file(path, name, api_key)
                if result.get("status") == "success":
                    indexed.append({"file": name, "chunks": result["chunks"]})
                else:
                    skipped.append({"file": name, "reason": result.get("message", "unknown")})

        return {
            "status": "success" if indexed else "error",
            "indexed": indexed,
            "skipped": skipped,
            "message": f"Indexed {len(indexed)} file(s), skipped {len(skipped)}",
        }

    def reindex_from_uploads(self, api_key: Optional[str]) -> Dict[str, Any]:
        """Wipe the index and rebuild it from the files already in uploads/.

        Used after an embedding-model change. Prefers the manifest so display
        names and de-duplication are exact; falls back to scanning the folder
        and stripping the "<stamp>_<id>_" prefix for indexes predating it.
        """
        self._require_admin(api_key)

        manifest = self._load_manifest()
        if manifest:
            plan = [
                (name, os.path.join(LOCAL_UPLOADS_DIR, entry["stored_path"]))
                for name, entry in sorted(manifest.items())
                if entry.get("stored_path")
            ]
        else:
            # Legacy layout: recover the display name and keep only the newest
            # copy of each, since the old delete bug left duplicates behind.
            newest: Dict[str, str] = {}
            for name in sorted(os.listdir(LOCAL_UPLOADS_DIR)):
                path = os.path.join(LOCAL_UPLOADS_DIR, name)
                if not os.path.isfile(path) or name.startswith("_"):
                    continue
                m = re.match(r"^\d{8}_\d{6}_[0-9a-f]{8}_(.+)$", name)
                display = m.group(1) if m else name
                newest[display] = path  # sorted order => last wins
            plan = sorted(newest.items())

        self.reset_index()

        rebuilt, failed = [], []
        new_manifest: Dict[str, Dict[str, Any]] = {}
        for display_name, path in plan:
            if not os.path.exists(path):
                failed.append({"file": display_name, "reason": "stored file missing"})
                continue
            content = self.extract_text(path)
            chunks = SimpleTextSplitter().split_text(content)
            if not chunks:
                failed.append({"file": display_name, "reason": "no extractable text"})
                continue
            stored_name = os.path.basename(path)
            uploaded_at = _utcnow()
            try:
                with self._lock:
                    self._add_chunks(
                        display_name,
                        chunks,
                        {"stored_path": stored_name, "uploaded_at": uploaded_at},
                    )
            except Exception as e:
                failed.append({"file": display_name, "reason": str(e)})
                continue
            rebuilt.append({"file": display_name, "chunks": len(chunks)})
            new_manifest[display_name] = {
                "stored_path": stored_name,
                "chunks": len(chunks),
                "chars": len(content),
                "uploaded_at": uploaded_at,
            }

        with self._lock:
            self._save_manifest(new_manifest)

        logger.info("Reindexed %d file(s), %d failed", len(rebuilt), len(failed))
        return {
            "status": "success",
            "rebuilt": rebuilt,
            "failed": failed,
            "message": f"Rebuilt {len(rebuilt)} file(s), {len(failed)} failed",
        }

    # ── retrieval ────────────────────────────────────────────────────────────

    def query(self, question: str, k: int = RETRIEVAL_K) -> Dict[str, Any]:
        """Retrieve relevant chunks.

        Returns ``{"status", "context", "sources"}``. ``status`` is one of
        ``ok`` (relevant context found), ``empty`` (index has no relevant
        match), ``unavailable`` (not initialised), ``stale`` (index built with
        a different embedding model) or ``error``.

        Callers must treat anything other than ``ok`` as "no context" -- the
        old string-sentinel protocol ("No results found", "Error: ...") was
        easy to mistake for real content.
        """
        if self.collection is None:
            return {"status": "unavailable", "context": "", "sources": []}
        if self.stale:
            return {"status": "stale", "context": "", "sources": []}
        if not (question or "").strip():
            return {"status": "empty", "context": "", "sources": []}

        try:
            if self.collection.count() == 0:
                return {"status": "empty", "context": "", "sources": []}

            res = self.collection.query(
                query_texts=[question],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
            documents = (res.get("documents") or [[]])[0]
            metadatas = (res.get("metadatas") or [[]])[0]
            distances = (res.get("distances") or [[]])[0]

            kept = []
            for doc, meta, dist in zip(documents, metadatas, distances):
                if dist is not None and dist > MAX_DISTANCE:
                    continue
                kept.append((doc, (meta or {}).get("source", "unknown"), dist))

            if not kept:
                best = min(distances) if distances else None
                logger.info(
                    "No chunk within distance %.2f for query %r (best=%s)",
                    MAX_DISTANCE,
                    question[:60],
                    f"{best:.3f}" if best is not None else "n/a",
                )
                return {"status": "empty", "context": "", "sources": []}

            context = "\n\n".join(f"[Source: {src}]\n{doc}" for doc, src, _ in kept)
            sources = sorted({src for _, src, _ in kept})
            logger.info(
                "Retrieved %d chunk(s) from %s (best distance %.3f)",
                len(kept),
                ", ".join(sources),
                kept[0][2] if kept[0][2] is not None else -1,
            )
            return {"status": "ok", "context": context, "sources": sources}
        except Exception as e:
            logger.error("Retrieval error: %s", e)
            return {"status": "error", "context": "", "sources": [], "message": str(e)}

    # ── inventory / maintenance ──────────────────────────────────────────────

    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        """Unique indexed sources with chunk counts."""
        if self.collection is None:
            return []
        try:
            data = self.collection.get(include=["metadatas"])
            sources: Dict[str, Dict[str, Any]] = {}
            for meta in data.get("metadatas") or []:
                if not meta:
                    continue
                src = meta.get("source", "unknown")
                entry = sources.setdefault(
                    src, {"source": src, "chunks": 0, "uploaded_at": meta.get("uploaded_at", "")}
                )
                entry["chunks"] += 1
            return sorted(sources.values(), key=lambda d: d["source"].lower())
        except Exception as e:
            logger.error("Error listing indexed documents: %s", e)
            return []

    def _remove_document_unlocked(self, source_name: str) -> Dict[str, Any]:
        if self.collection is None:
            return {"status": "error", "message": "Knowledge base not initialised"}

        removed_file = False
        try:
            self.collection.delete(where={"source": source_name})
        except Exception as e:
            logger.error("Error deleting chunks for %s: %s", source_name, e)
            return {"status": "error", "message": str(e)}

        manifest = self._load_manifest()
        entry = manifest.pop(source_name, None)
        if entry and entry.get("stored_path"):
            path = os.path.join(LOCAL_UPLOADS_DIR, entry["stored_path"])
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    removed_file = True
            except OSError as e:
                logger.warning("Could not delete stored file %s: %s", path, e)
        self._save_manifest(manifest)

        return {
            "status": "success",
            "source": source_name,
            "file_removed": removed_file,
            "message": f"Removed {source_name} from the organisation context",
        }

    def remove_document(self, source_name: str) -> Dict[str, Any]:
        with self._lock:
            result = self._remove_document_unlocked(source_name)
        if result.get("status") == "success":
            logger.info("Removed %s (file_removed=%s)", source_name, result["file_removed"])
        return result

    def prune_orphans(self) -> Dict[str, Any]:
        """Delete files in uploads/ that no manifest entry references."""
        manifest = self._load_manifest()
        keep = {e.get("stored_path") for e in manifest.values() if e.get("stored_path")}
        removed, freed = [], 0
        for name in sorted(os.listdir(LOCAL_UPLOADS_DIR)):
            path = os.path.join(LOCAL_UPLOADS_DIR, name)
            if not os.path.isfile(path) or name.startswith("_") or name in keep:
                continue
            try:
                freed += os.path.getsize(path)
                os.remove(path)
                removed.append(name)
            except OSError as e:
                logger.warning("Could not prune %s: %s", path, e)
        return {"status": "success", "removed": removed, "freed_bytes": freed}

    def get_context_status(self) -> Dict[str, Any]:
        docs = self.list_indexed_documents()
        storage_bytes = 0
        try:
            for name in os.listdir(LOCAL_UPLOADS_DIR):
                path = os.path.join(LOCAL_UPLOADS_DIR, name)
                if os.path.isfile(path) and not name.startswith("_"):
                    storage_bytes += os.path.getsize(path)
        except OSError:
            pass

        return {
            "ready": self.ready,
            "stale": self.stale,
            "init_error": self.init_error,
            "embedding_model": EMBEDDING_ID,
            "total_files": len(docs),
            "total_chunks": sum(d["chunks"] for d in docs),
            "storage_bytes": storage_bytes,
            "uploads_dir": LOCAL_UPLOADS_DIR,
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
            "max_upload_bytes": MAX_UPLOAD_BYTES,
        }


kb = KnowledgeBase()
