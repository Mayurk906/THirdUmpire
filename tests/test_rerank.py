from __future__ import annotations

from src.rerank import _sigmoid, rerank
from src.schemas import Chunk, ScoredChunk


def make_scored(chunk_id: str, text: str, first_stage_score: float) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        body="laws",
        competition="laws",
        gender="all",
        doc_type="laws",
        effective_from="2026",
        law_no=1,
        clause="1.1",
        clause_title="Title",
        section_path="1 > 1.1",
        page_start=1,
        page_end=1,
        chunk_index=0,
        content_hash="hash",
        status="active",
        text=text,
    )
    return ScoredChunk(chunk=chunk, score=first_stage_score)


class _FakeCrossEncoder:
    """Returns a fixed raw logit per chunk_id, keyed by order of `.predict()` calls."""

    def __init__(self, logits_by_text: dict[str, float]):
        self._logits = logits_by_text

    def predict(self, pairs):
        return [self._logits[text] for _query, text in pairs]


def test_blend_alpha_1_matches_pure_rerank_score(monkeypatch):
    candidates = [make_scored("a", "a", 0.9), make_scored("b", "b", 0.5)]
    fake = _FakeCrossEncoder({"a": -1.0, "b": 1.0})  # reranker flips the order
    monkeypatch.setattr("src.rerank._get_cross_encoder", lambda reranker: fake)

    result = rerank("q", candidates, blend_alpha=1.0)

    assert [sc.chunk.chunk_id for sc in result] == ["b", "a"]
    assert result[0].score == _sigmoid(1.0)


def test_blend_can_preserve_first_stage_top_rank(monkeypatch):
    # First-stage rank 0 ("a") vs rank 1 ("b"); reranker only barely prefers
    # "b" (tiny logit gap) -- a low blend_alpha should keep "a" on top,
    # recovering the Recall@1 the pure-reranker ordering gave up.
    candidates = [make_scored("a", "a", 0.9), make_scored("b", "b", 0.8)]
    fake = _FakeCrossEncoder({"a": 0.01, "b": 0.02})
    monkeypatch.setattr("src.rerank._get_cross_encoder", lambda reranker: fake)

    result = rerank("q", candidates, blend_alpha=0.5)

    assert result[0].chunk.chunk_id == "a"


def test_rerank_still_handles_empty_candidates():
    assert rerank("q", []) == []


def test_raw_score_is_unblended_cross_encoder_logit(monkeypatch):
    candidates = [make_scored("a", "a", 0.9)]
    fake = _FakeCrossEncoder({"a": 2.5})
    monkeypatch.setattr("src.rerank._get_cross_encoder", lambda reranker: fake)

    result = rerank("q", candidates, blend_alpha=0.7)

    assert result[0].raw_score == 2.5
