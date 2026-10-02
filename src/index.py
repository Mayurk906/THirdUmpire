"""Incremental embed + upsert of data/processed/chunks.jsonl into Chroma.

Only chunks that are new or whose content_hash changed get re-embedded;
unchanged chunks are skipped. Chunks belonging to a re-ingested document
whose chunk_id is no longer produced are deleted (stale removal). After
syncing, INDEX_VERSION (sha256 of every active chunk_id:content_hash pair)
is written to chroma_db/index_version.txt.

Usage: python -m src.index [doc_id ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, Iterable

import chromadb

from src.config import BASE_DIR, CHROMA_DIR, COLLECTION, EMBED_MODEL, configure_logging
from src.schemas import Chunk

logger = logging.getLogger(__name__)

PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

EmbedFn = Callable[[list[str]], list[list[float]]]


def load_chunks(doc_ids: list[str] | None = None) -> list[Chunk]:
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            chunk = Chunk(**json.loads(line))
            if doc_ids is None or chunk.doc_id in doc_ids:
                chunks.append(chunk)
    return chunks


def get_embedder():
    """Return the embedding model itself (not just embed_documents), since
    retrieve.py also needs embed_query (which applies BGE_QUERY_INSTRUCTION
    differently than document embedding)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})


def get_collection(chroma_dir: str | None = None):
    client = chromadb.PersistentClient(path=chroma_dir or str(BASE_DIR / CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION)


def existing_metadata(collection, doc_ids: Iterable[str]) -> dict[str, dict]:
    doc_ids = list(doc_ids)
    if not doc_ids:
        return {}
    result = collection.get(where={"doc_id": {"$in": doc_ids}}, include=["metadatas"])
    return dict(zip(result["ids"], result["metadatas"]))


def existing_hashes(collection, doc_ids: Iterable[str]) -> dict[str, str]:
    return {chunk_id: meta["content_hash"] for chunk_id, meta in existing_metadata(collection, doc_ids).items()}


def sync_index(chunks: list[Chunk], collection, embed_fn: EmbedFn) -> dict[str, int]:
    doc_ids = sorted({c.doc_id for c in chunks})
    existing = existing_metadata(collection, doc_ids)

    current_ids = {c.chunk_id for c in chunks}
    to_embed = [c for c in chunks if c.chunk_id not in existing or existing[c.chunk_id]["content_hash"] != c.content_hash]
    to_embed_ids = {c.chunk_id for c in to_embed}

    # A chunk whose clause text (and therefore content_hash) is unchanged can
    # still need its metadata refreshed -- e.g. scraper/supersede.py flips
    # status from "active" to "superseded" without touching the clause text,
    # and a hash-only comparison would otherwise leave that stale "active"
    # status in Chroma forever. Update those in place with no re-embedding.
    to_update_status = [
        c
        for c in chunks
        if c.chunk_id not in to_embed_ids and c.chunk_id in existing and existing[c.chunk_id].get("status") != c.status
    ]
    skipped = len(chunks) - len(to_embed) - len(to_update_status)

    if to_embed:
        embeddings = embed_fn([c.text for c in to_embed])
        collection.upsert(
            ids=[c.chunk_id for c in to_embed],
            embeddings=embeddings,
            documents=[c.text for c in to_embed],
            metadatas=[c.metadata() for c in to_embed],
        )

    if to_update_status:
        collection.update(ids=[c.chunk_id for c in to_update_status], metadatas=[c.metadata() for c in to_update_status])

    stale_ids = set(existing.keys()) - current_ids
    if stale_ids:
        collection.delete(ids=list(stale_ids))

    return {
        "embedded": len(to_embed),
        "skipped": skipped,
        "deleted": len(stale_ids),
        "status_updated": len(to_update_status),
    }


def compute_index_version(collection) -> str:
    result = collection.get(where={"status": "active"}, include=["metadatas"])
    pairs = sorted(f"{id_}:{meta['content_hash']}" for id_, meta in zip(result["ids"], result["metadatas"]))
    return hashlib.sha256("\n".join(pairs).encode("utf-8")).hexdigest()


def write_index_version(version: str, chroma_dir: str | None = None) -> None:
    path = Path(chroma_dir) if chroma_dir is not None else BASE_DIR / CHROMA_DIR
    path.mkdir(parents=True, exist_ok=True)
    (path / "index_version.txt").write_text(version, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("doc_ids", nargs="*", help="doc_ids to (re-)index (default: every doc in chunks.jsonl)")
    args = parser.parse_args()

    configure_logging()

    if not CHUNKS_PATH.exists():
        logger.error("%s not found; run 'python -m src.chunk' first", CHUNKS_PATH)
        return 1

    chunks = load_chunks(args.doc_ids or None)
    if not chunks:
        logger.error("No chunks found for doc_ids=%s", args.doc_ids or "<all>")
        return 1

    logger.info("Loading embedder %s ...", EMBED_MODEL)
    model = get_embedder()
    collection = get_collection()

    stats = sync_index(chunks, collection, model.embed_documents)
    version = compute_index_version(collection)
    write_index_version(version)

    logger.info(
        "Indexed %d chunks (embedded %d, skipped %d, deleted %d) -> INDEX_VERSION=%s",
        len(chunks),
        stats["embedded"],
        stats["skipped"],
        stats["deleted"],
        version,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
