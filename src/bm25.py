"""BM25 (lexical) first-stage retrieval (PROJECT_PLAN.md Phase 5).

Scores the WHOLE corpus so BM25's IDF statistics aren't distorted by a
competition/status filter, then applies the QueryPlan's filter afterwards
when selecting the top k -- filtering before scoring would make a term's
"rarity" depend on which competition happened to be asked about, and
applying the filter only for display (not for selection) is exactly the
"filters silently bypassed" bug class PROJECT_PLAN.md calls out for this
phase, so every result here is checked against the plan before it can be
returned.

The index is rebuilt from Chroma whenever INDEX_VERSION changes (cached
by that version string), matching the plan's "rebuild or reload ... when
INDEX_VERSION changes".
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import NamedTuple

from rank_bm25 import BM25Okapi

from src.config import BASE_DIR, CHROMA_DIR
from src.schemas import Chunk, QueryPlan, ScoredChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _Bm25Index(NamedTuple):
    chunks: list[Chunk]
    bm25: BM25Okapi


def _load_all_chunks() -> list[Chunk]:
    from src.index import get_collection

    collection = get_collection()
    result = collection.get(include=["documents", "metadatas"])
    return [Chunk(text=document, **metadata) for document, metadata in zip(result["documents"], result["metadatas"])]


def _build_index(chunks: list[Chunk]) -> _Bm25Index:
    return _Bm25Index(chunks=chunks, bm25=BM25Okapi([tokenize(c.text) for c in chunks]))


@lru_cache(maxsize=1)
def _get_index(index_version: str) -> _Bm25Index:
    """`index_version` is only a cache key (lru_cache(maxsize=1) means a new
    version evicts the old one); the chunks themselves always come fresh
    from Chroma."""
    return _build_index(_load_all_chunks())


def current_index_version() -> str:
    path = BASE_DIR / CHROMA_DIR / "index_version.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else "unversioned"


def matches_filter(chunk: Chunk, plan: QueryPlan) -> bool:
    if chunk.status != "active":
        return False
    if plan.allowed_competitions and chunk.competition not in plan.allowed_competitions:
        return False
    return True


def retrieve_bm25(plan: QueryPlan, k: int, index: _Bm25Index | None = None) -> list[ScoredChunk]:
    """`index` is exposed for tests (an in-memory index, skipping Chroma);
    callers otherwise get the live, version-cached index."""
    if index is None:
        index = _get_index(current_index_version())

    scores = index.bm25.get_scores(tokenize(plan.question))
    scored = [
        ScoredChunk(chunk=chunk, score=float(score))
        for chunk, score in zip(index.chunks, scores)
        if matches_filter(chunk, plan)
    ]
    scored.sort(key=lambda sc: sc.score, reverse=True)
    return scored[:k]
