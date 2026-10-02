"""Reciprocal Rank Fusion of ranked chunk lists (PROJECT_PLAN.md Phase 5).

A chunk's fused score is the sum, over every input list it appears in, of
1 / (rrf_k + rank) where rank is 1-based within that list; a chunk missing
from a list contributes nothing from it. Lists are merged by chunk_id
(not object identity), since the same logical chunk is a distinct
ScoredChunk instance in the dense and BM25 results.
"""
from __future__ import annotations

from src.config import RRF_K
from src.schemas import Chunk, ScoredChunk


def reciprocal_rank_fusion(ranked_lists: list[list[ScoredChunk]], k: int, rrf_k: int = RRF_K) -> list[ScoredChunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, scored_chunk in enumerate(ranked, start=1):
            chunk_id = scored_chunk.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            chunks.setdefault(chunk_id, scored_chunk.chunk)

    fused = [ScoredChunk(chunk=chunks[chunk_id], score=score) for chunk_id, score in scores.items()]
    fused.sort(key=lambda sc: sc.score, reverse=True)
    return fused[:k]
