"""Pure metric functions for the eval harness.

Dependency-free and unit-tested on hand-made toy data (PROJECT_PLAN.md
Phase 3) so the scoring logic can be trusted independently of retrieval,
Chroma or the LLM.
"""
from __future__ import annotations


def hit_rank(ranked: list[tuple[str, str]], gold: set[tuple[str, str]]) -> int | None:
    """1-based rank of the first item in `ranked` that's in `gold`, or None if none match."""
    for i, item in enumerate(ranked, start=1):
        if item in gold:
            return i
    return None


def recall_at_k(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None, k: int = 10) -> float:
    if rank is not None and rank <= k:
        return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy's default 'linear' method). p in [0, 100]."""
    if not values:
        raise ValueError("percentile() requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac
