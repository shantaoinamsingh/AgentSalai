#!/usr/bin/env python3
"""Manage the organisation knowledge base from the command line.

    python manage_kb.py status
    python manage_kb.py index --folder ./docs
    python manage_kb.py reindex          # rebuild from uploads/ (after a model change)
    python manage_kb.py reset            # drop every vector
    python manage_kb.py prune            # delete unreferenced files in uploads/
    python manage_kb.py query "some question"

`reindex` is the one to run after upgrading from the legacy embedding scheme:
the old index stored LLM-hallucinated vectors, so it must be rebuilt before
retrieval returns anything meaningful.
"""
import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(levelname)s: %(message)s")

from knowledge_base import kb  # noqa: E402  (import after load_dotenv)


def _require_key(args) -> str:
    key = args.api_key or os.getenv("ADMIN_API_KEY")
    if not key:
        sys.exit("ADMIN_API_KEY is not set. Put it in .env or pass --api-key.")
    return key


def cmd_status(_args) -> int:
    s = kb.get_context_status()
    if s["init_error"]:
        print(f"Knowledge base failed to initialise: {s['init_error']}")
        return 1
    if s["stale"]:
        print("Index is STALE - it was built with a different embedding model.")
        print("Retrieval is disabled. Rebuild it with:  python manage_kb.py reindex")
        return 1

    print(f"Ready         : {s['ready']}")
    print(f"Embedding     : {s['embedding_model']}")
    print(f"Files indexed : {s['total_files']}")
    print(f"Chunks        : {s['total_chunks']}")
    print(f"Upload storage: {s['storage_bytes'] / 1e6:.1f} MB in {s['uploads_dir']}")
    for doc in kb.list_indexed_documents():
        print(f"  - {doc['source']}  ({doc['chunks']} chunks)")
    return 0


def cmd_index(args) -> int:
    key = _require_key(args)
    if not args.folder:
        sys.exit("--folder is required for 'index'")
    result = kb.index_documents(args.folder, key)
    for item in result.get("indexed", []):
        print(f"  indexed {item['file']} ({item['chunks']} chunks)")
    for item in result.get("skipped", []):
        print(f"  skipped {item['file']}: {item['reason']}")
    print(result["message"])
    return 0 if result.get("status") == "success" else 1


def cmd_reindex(args) -> int:
    key = _require_key(args)
    print("Rebuilding the index from uploads/ ...")
    result = kb.reindex_from_uploads(key)
    for item in result.get("rebuilt", []):
        print(f"  rebuilt {item['file']} ({item['chunks']} chunks)")
    for item in result.get("failed", []):
        print(f"  FAILED  {item['file']}: {item['reason']}")
    print(result["message"])
    return 0


def cmd_reset(args) -> int:
    _require_key(args)
    if not args.yes:
        confirm = input("Delete every indexed vector? The uploaded files stay. [y/N] ")
        if confirm.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1
    kb.reset_index()
    print("Index reset. Run 'python manage_kb.py reindex' to rebuild from uploads/.")
    return 0


def cmd_prune(args) -> int:
    _require_key(args)
    result = kb.prune_orphans()
    for name in result["removed"]:
        print(f"  removed {name}")
    print(f"Pruned {len(result['removed'])} file(s), freed {result['freed_bytes'] / 1e6:.1f} MB")
    return 0


def cmd_query(args) -> int:
    result = kb.query(args.text)
    print(f"status: {result['status']}")
    if result["status"] != "ok":
        print("(no relevant context - the model would answer from general knowledge)")
        return 0
    print(f"sources: {', '.join(result['sources'])}\n")
    print(result["context"][:2000])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage the Salai organisation knowledge base")
    parser.add_argument("--api-key", default=None, help="admin key (defaults to $ADMIN_API_KEY)")
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("status", help="show index contents")

    p_index = sub.add_parser("index", help="index every supported file in a folder")
    p_index.add_argument("--folder", required=True)

    sub.add_parser("reindex", help="wipe and rebuild the index from uploads/")

    p_reset = sub.add_parser("reset", help="delete every vector")
    p_reset.add_argument("-y", "--yes", action="store_true", help="skip confirmation")

    sub.add_parser("prune", help="delete uploaded files no longer in the index")

    p_query = sub.add_parser("query", help="test retrieval for a question")
    p_query.add_argument("text")

    args = parser.parse_args()
    handlers = {
        "status": cmd_status,
        "index": cmd_index,
        "reindex": cmd_reindex,
        "reset": cmd_reset,
        "prune": cmd_prune,
        "query": cmd_query,
    }
    try:
        return handlers[args.action](args)
    except PermissionError:
        sys.exit("Invalid admin API key.")
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    sys.exit(main())
