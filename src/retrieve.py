"""Dense retrieval (first stage) with the query plan's metadata filter.

`first_stage` is a string switch ("dense" now, "hybrid" in Phase 5) so later
phases can plug in a new first stage without changing callers -- see
PROJECT_PLAN.md Phase 2 step 2 and Phase 5.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.config import BGE_QUERY_INSTRUCTION, RETRIEVE_K
from src.schemas import Chunk, QueryPlan, ScoredChunk


@lru_cache(maxsize=1)
def _get_embedder():
    from src.index import get_embedder

    return get_embedder()


@lru_cache(maxsize=1)
def _get_collection():
    from src.index import get_collection

    return get_collection()


def build_where(plan: QueryPlan) -> dict[str, Any]:
    conditions: list[dict[str, Any]] = [{"status": {"$eq": "active"}}]
    if plan.allowed_competitions:
        conditions.append({"competition": {"$in": plan.allowed_competitions}})
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _distance_to_similarity(distance: float) -> float:
    """Convert Chroma's squared-L2 distance (unit vectors) to cosine similarity."""
    return 1.0 - distance / 2.0


def retrieve_dense(plan: QueryPlan, k: int = RETRIEVE_K) -> list[ScoredChunk]:
    model = _get_embedder()
    query_text = f"{BGE_QUERY_INSTRUCTION}{plan.question}" if BGE_QUERY_INSTRUCTION else plan.question
    vector = model.embed_query(query_text)

    collection = _get_collection()
    result = collection.query(
        query_embeddings=[vector],
        n_results=k,
        where=build_where(plan),
        include=["documents", "metadatas", "distances"],
    )

    scored: list[ScoredChunk] = []
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    for document, metadata, distance in zip(documents, metadatas, distances):
        # metadata already includes chunk_id (Chunk.metadata() keeps every
        # field except text), so it must not also be passed positionally.
        chunk = Chunk(text=document, **metadata)
        scored.append(ScoredChunk(chunk=chunk, score=_distance_to_similarity(distance)))
    return scored


def retrieve_hybrid(plan: QueryPlan, k: int = RETRIEVE_K) -> list[ScoredChunk]:
    from src.bm25 import retrieve_bm25
    from src.fusion import reciprocal_rank_fusion

    dense = retrieve_dense(plan, k)
    lexical = retrieve_bm25(plan, k)
    return reciprocal_rank_fusion([dense, lexical], k=k)


def retrieve(plan: QueryPlan, k: int = RETRIEVE_K, first_stage: str = "dense") -> list[ScoredChunk]:
    if first_stage == "dense":
        return retrieve_dense(plan, k)
    if first_stage == "hybrid":
        return retrieve_hybrid(plan, k)
    raise NotImplementedError(f"first_stage={first_stage!r} isn't implemented (expected 'dense' or 'hybrid')")
