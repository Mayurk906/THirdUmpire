"""Orchestrates parse -> chunk -> index, and prints a summary.

Usage: python -m src.ingest [doc_id ...]
"""
from __future__ import annotations

import argparse
import logging

from src.chunk import CHUNKS_PATH, PROCESSED_DIR, chunk_document, load_sources
from src.config import configure_logging
from src.index import (
    compute_index_version,
    get_collection,
    get_embedder,
    load_chunks,
    sync_index,
    write_index_version,
)
from src.parse import RAW_DIR, parse_doc, save_pages
from src.schemas import Chunk

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("doc_ids", nargs="*", help="doc_ids to ingest (default: every doc in sources.yaml)")
    args = parser.parse_args()

    configure_logging()
    sources = load_sources()
    doc_ids = args.doc_ids or list(sources.keys())

    for doc_id in doc_ids:
        if doc_id not in sources:
            logger.error("%s: not found in sources.yaml", doc_id)
            return 1

    # --- parse ---
    logger.info("=== Parsing %d document(s) ===", len(doc_ids))
    page_counts: dict[str, int] = {}
    for doc_id in doc_ids:
        pdf_path = RAW_DIR / f"{doc_id}.pdf"
        if not pdf_path.exists():
            logger.error("%s: %s not found; run 'python -m src.download' first", doc_id, pdf_path)
            return 1
        records = parse_doc(doc_id)
        save_pages(doc_id, records)
        page_counts[doc_id] = len(records)
        logger.info("%s: %d pages parsed", doc_id, len(records))

    # --- chunk ---
    logger.info("=== Chunking %d document(s) ===", len(doc_ids))
    all_chunks: list[Chunk] = load_chunks_excluding(doc_ids)
    clause_counts: dict[str, int] = {}
    chunk_counts: dict[str, int] = {}
    for doc_id in doc_ids:
        chunks = chunk_document(doc_id, sources[doc_id])
        clause_counts[doc_id] = len({c.clause for c in chunks})
        chunk_counts[doc_id] = len(chunks)
        all_chunks.extend(chunks)
        logger.info("%s: %d clauses -> %d chunks", doc_id, clause_counts[doc_id], chunk_counts[doc_id])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.model_dump_json() + "\n")

    # --- index ---
    logger.info("=== Indexing %d document(s) ===", len(doc_ids))
    chunks_to_index = load_chunks(doc_ids)
    logger.info("Loading embedder ...")
    model = get_embedder()
    collection = get_collection()
    stats = sync_index(chunks_to_index, collection, model.embed_documents)
    version = compute_index_version(collection)
    write_index_version(version)

    # --- summary ---
    lengths = [len(c.text) for c in chunks_to_index]
    logger.info("=== Summary ===")
    for doc_id in doc_ids:
        logger.info(
            "%s: %d pages, %d clauses, %d chunks",
            doc_id,
            page_counts[doc_id],
            clause_counts[doc_id],
            chunk_counts[doc_id],
        )
    logger.info(
        "Chunks embedded: %d, status-only updated: %d, skipped (unchanged): %d, deleted (stale): %d",
        stats["embedded"],
        stats["status_updated"],
        stats["skipped"],
        stats["deleted"],
    )
    if lengths:
        logger.info("Chunk length: avg %.0f chars, max %d chars", sum(lengths) / len(lengths), max(lengths))
    logger.info("INDEX_VERSION=%s", version)
    return 0


def load_chunks_excluding(doc_ids: list[str]) -> list[Chunk]:
    """Chunks already on disk for docs NOT in this ingest run, so re-writing
    chunks.jsonl for a partial re-ingest doesn't drop the other documents."""
    if not CHUNKS_PATH.exists():
        return []
    existing = load_chunks(doc_ids=None)
    return [c for c in existing if c.doc_id not in doc_ids]


if __name__ == "__main__":
    raise SystemExit(main())
