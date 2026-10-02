from __future__ import annotations

from src import pipeline
from src.pipeline import select_context, top_up_compared
from src.query_analyzer import build_query_plan
from src.schemas import Chunk, QueryPlan, ScoredChunk


def make_scored(chunk_id: str, competition: str, gender: str = "men", score: float = 0.5) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        body="icc",
        competition=competition,
        gender=gender,
        doc_type="playing_conditions",
        effective_from="2025",
        law_no=-1,
        clause="12.1",
        clause_title="Minimum Over Rates",
        section_path="12 > 12.1",
        page_start=1,
        page_end=1,
        chunk_index=0,
        content_hash="hash",
        status="active",
        text="Minimum over rate.",
    )
    return ScoredChunk(chunk=chunk, score=score)


COMPARISON = "How does the minimum over rate differ between Test, ODI and T20I cricket?"


def ids(chunks: list[Chunk]) -> list[str]:
    return [c.chunk_id for c in chunks]


def test_select_context_without_comparison_is_plain_top_k():
    reranked = [make_scored(str(i), "icc_t20i") for i in range(8)]
    assert ids(select_context(reranked, 5)) == ["0", "1", "2", "3", "4"]


def test_select_context_guarantees_each_compared_competition():
    reranked = [
        make_scored("t20_m", "icc_t20i"),
        make_scored("t20_w", "icc_t20i", "women"),
        make_scored("odi_w", "icc_odi", "women"),
        make_scored("dom_t20_m", "bcci_domestic_t20"),
        make_scored("dom_t20_w", "bcci_domestic_t20", "women"),
        make_scored("odi_m", "icc_odi"),
        make_scored("test_w", "icc_test", "women"),
        make_scored("test_m", "icc_test"),
    ]
    context = ids(select_context(reranked, 5, ["icc_odi", "icc_t20i", "icc_test"], COMPARISON))
    assert len(context) == 5
    # Men's chunks preferred for an unqualified question, kept in rank order.
    assert context == ["t20_m", "t20_w", "odi_w", "odi_m", "test_m"]


def test_select_context_prefers_women_when_asked():
    reranked = [
        make_scored("odi_m", "icc_odi"),
        make_scored("t20_m", "icc_t20i"),
        make_scored("filler", "laws", "all"),
        make_scored("test_m", "icc_test"),
        make_scored("test_w", "icc_test", "women"),
    ]
    context = ids(select_context(reranked, 3, ["icc_odi", "icc_test"], "women's Test vs ODI over rate"))
    assert "test_w" in context and "test_m" not in context
    # Falls back to the other gender when the preferred one is absent.
    assert "odi_m" in context


def test_select_context_ignores_compared_competition_with_no_chunks():
    reranked = [make_scored(str(i), "icc_t20i") for i in range(6)]
    context = ids(select_context(reranked, 5, ["icc_odi", "icc_t20i"], COMPARISON))
    assert context == ["0", "1", "2", "3", "4"]


def test_top_up_fetches_only_missing_competitions(monkeypatch):
    calls: list[QueryPlan] = []

    def fake_retrieve(plan, k, first_stage):
        calls.append(plan)
        comp = plan.allowed_competitions[0]
        return [make_scored(f"{comp}_new", comp), make_scored("t20_m", "icc_t20i")]

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    plan = build_query_plan(COMPARISON)
    retrieved = [make_scored("t20_m", "icc_t20i"), make_scored("odi_w", "icc_odi", "women")]

    result = top_up_compared(plan, retrieved, "dense")

    assert [p.allowed_competitions for p in calls] == [["icc_test"]]
    # Duplicate chunk_ids are not re-added.
    assert ids([sc.chunk for sc in result]) == ["t20_m", "odi_w", "icc_test_new"]
