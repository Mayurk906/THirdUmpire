"""Cross-encoder reranking (base or fine-tuned).

Scores every (query, chunk_text) pair, sorts by raw score descending, and
returns ALL candidates with both the raw score and a sigmoid-normalised
score in [0, 1]. The caller (pipeline.py) takes the top CONTEXT_K.
"""
from __future__ import annotations

import math
from functools import lru_cache

from src.config import RERANKER_FT_PATH, RERANKER_MODEL
from src.schemas import ScoredChunk


@lru_cache(maxsize=2)
def _get_cross_encoder(reranker: str):
    from sentence_transformers import CrossEncoder

    if reranker == "base":
        return CrossEncoder(RERANKER_MODEL)
    if reranker == "ft":
        from pathlib import Path

        path = Path(RERANKER_FT_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Fine-tuned reranker not found at {path} (it's trained in Phase 6). Use reranker='base' until then."
            )
        return CrossEncoder(str(path))
    raise ValueError(f"reranker must be 'base' or 'ft', got {reranker!r}")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rerank(query: str, candidates: list[ScoredChunk], reranker: str = "base") -> list[ScoredChunk]:
    if not candidates:
        return []

    model = _get_cross_encoder(reranker)
    pairs = [(query, c.chunk.text) for c in candidates]
    raw_scores = model.predict(pairs)

    scored = [
        ScoredChunk(chunk=c.chunk, score=_sigmoid(float(s)), raw_score=float(s))
        for c, s in zip(candidates, raw_scores)
    ]
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored
