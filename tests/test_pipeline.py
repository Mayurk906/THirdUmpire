from __future__ import annotations

from src import answer_cache, pipeline
from src.pipeline import select_context, top_up_compared
from src.query_analyzer import build_query_plan
from src.schemas import Answer, Chunk, QueryPlan, ScoredChunk


def make_scored(
    chunk_id: str,
    competition: str,
    gender: str = "men",
    score: float = 0.5,
    clause_title: str = "Minimum Over Rates",
) -> ScoredChunk:
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
        clause_title=clause_title,
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


def test_select_context_caps_duplicate_competitions_even_without_comparison():
    # No compared_competitions detected, but the top 5 by score are still 5
    # near-identical "Minimum Over Rates" copies from different competitions
    # -- the general diversity cap (not just the comparison-guarantee path)
    # must kick in here too, or a plain single-format question suffers the
    # same crowding bug a comparison question does.
    reranked = [
        make_scored("t20i_m", "icc_t20i", score=0.95),
        make_scored("t20i_w", "icc_t20i", "women", score=0.94),
        make_scored("odi_w", "icc_odi", "women", score=0.93),
        make_scored("dom_t20_m", "bcci_domestic_t20", score=0.92),
        make_scored("dom_t20_w", "bcci_domestic_t20", "women", score=0.91),
        make_scored("real_answer", "laws", score=0.80, clause_title="Follow-on"),
    ]
    context = ids(select_context(reranked, 5))
    assert "real_answer" in context


def test_select_context_caps_are_case_insensitive_on_clause_title():
    # Found live: ICC men's T20I/ODI docs title this clause "Minimum over
    # rates" (lowercase "over"); every other competition uses "Minimum Over
    # Rates". A case-sensitive cap key would treat these as two separate
    # clauses and let twice as many near-duplicates through.
    reranked = [
        make_scored("t20i_m", "icc_t20i", score=0.95, clause_title="Minimum over rates"),
        make_scored("odi_m", "icc_odi", score=0.94, clause_title="Minimum over rates"),
        make_scored("t20i_w", "icc_t20i", "women", score=0.93, clause_title="Minimum Over Rates"),
        make_scored("odi_w", "icc_odi", "women", score=0.92, clause_title="Minimum Over Rates"),
        make_scored("real_answer", "laws", score=0.80, clause_title="Follow-on"),
    ]
    context = ids(select_context(reranked, 3))
    assert "real_answer" in context


def test_select_context_same_clause_multi_chunk_is_never_capped():
    # Two chunks of the SAME (competition, gender)'s long clause, split by
    # chunk_index, must both survive -- they're pieces of one answer, not
    # duplicates, even though they share clause_title and competition.
    reranked = [
        make_scored("law_4_1_piece_a", "laws", score=0.9),
        make_scored("law_4_1_piece_b", "laws", score=0.89),
        make_scored("other_x", "icc_t20i", score=0.5),
        make_scored("other_y", "icc_odi", score=0.4),
        make_scored("other_z", "bcci_domestic_t20", score=0.3),
    ]
    context = ids(select_context(reranked, 2))
    assert context == ["law_4_1_piece_a", "law_4_1_piece_b"]


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


def test_run_query_second_call_hits_the_answer_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(answer_cache, "CACHE_PATH", tmp_path / "answers.sqlite3")
    monkeypatch.setattr(answer_cache, "_index_version", lambda: "v1")

    candidates = [make_scored("laws_1_1", "laws", clause_title="Preamble")]
    monkeypatch.setattr(pipeline, "retrieve", lambda plan, k, first_stage: candidates)
    monkeypatch.setattr(pipeline, "rerank", lambda question, retrieved, reranker: retrieved)

    call_count = 0

    def fake_generate_answer(question, context_chunks):
        nonlocal call_count
        call_count += 1
        return Answer(answer="Eleven players.", found=True, applies_to="laws", citations=[])

    monkeypatch.setattr(pipeline, "generate_answer", fake_generate_answer)

    first = pipeline.run_query("How many players are in a team?")
    second = pipeline.run_query("How many players are in a team?")

    assert call_count == 1  # generate_answer only ran once; the 2nd call hit the cache.
    assert first.answer_cached is False
    assert second.answer_cached is True
    assert second.answer.answer == "Eleven players."


def test_run_query_cache_disabled_calls_generate_answer_every_time(monkeypatch, tmp_path):
    monkeypatch.setattr(answer_cache, "CACHE_PATH", tmp_path / "answers.sqlite3")
    monkeypatch.setattr(pipeline, "ANSWER_CACHE_ENABLED", False)

    candidates = [make_scored("laws_1_1", "laws", clause_title="Preamble")]
    monkeypatch.setattr(pipeline, "retrieve", lambda plan, k, first_stage: candidates)
    monkeypatch.setattr(pipeline, "rerank", lambda question, retrieved, reranker: retrieved)

    call_count = 0

    def fake_generate_answer(question, context_chunks):
        nonlocal call_count
        call_count += 1
        return Answer(answer="x", found=True, applies_to="laws", citations=[])

    monkeypatch.setattr(pipeline, "generate_answer", fake_generate_answer)

    pipeline.run_query("Same question?")
    pipeline.run_query("Same question?")

    assert call_count == 2
