from src.bm25 import _build_index, matches_filter, retrieve_bm25, tokenize
from src.schemas import Chunk, QueryPlan


def make_chunk(chunk_id: str, competition: str, status: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        body="mcc",
        competition=competition,
        gender="all",
        doc_type="laws",
        effective_from="2026-01-01",
        law_no=1,
        clause="1.1",
        clause_title="Title",
        section_path="LAW 1 > 1.1",
        page_start=1,
        page_end=1,
        chunk_index=0,
        content_hash="hash",
        status=status,
        text=text,
    )


CORPUS = [
    make_chunk("laws_bowled", "laws", "active", "A batter is out Bowled if the ball hits the wicket."),
    make_chunk("ipl_bowled", "ipl", "active", "A batter is out Bowled per IPL playing conditions."),
    make_chunk("icc_t20i_bowled", "icc_t20i", "active", "A batter is out Bowled per ICC T20I conditions."),
    make_chunk("laws_caught_old", "laws", "superseded", "A batter is out Caught under the old edition."),
    make_chunk("laws_caught", "laws", "active", "A batter is out Caught if the ball is held before touching ground."),
]


def index():
    return _build_index(CORPUS)


def test_tokenize_lowercases_and_splits_on_non_alnum() -> None:
    assert tokenize("Out Caught! (Law 33)") == ["out", "caught", "law", "33"]


def test_matches_filter_excludes_superseded() -> None:
    plan = QueryPlan(question="x", competition=None, allowed_competitions=None)
    chunk = make_chunk("a", "laws", "superseded", "text")
    assert matches_filter(chunk, plan) is False


def test_matches_filter_excludes_disallowed_competition() -> None:
    plan = QueryPlan(question="x", competition="ipl", allowed_competitions=["ipl", "laws"])
    chunk = make_chunk("a", "icc_t20i", "active", "text")
    assert matches_filter(chunk, plan) is False


def test_matches_filter_allows_active_in_allowed_competition() -> None:
    plan = QueryPlan(question="x", competition="ipl", allowed_competitions=["ipl", "laws"])
    chunk = make_chunk("a", "laws", "active", "text")
    assert matches_filter(chunk, plan) is True


def test_matches_filter_no_competition_filter_allows_any_active_competition() -> None:
    plan = QueryPlan(question="x", competition=None, allowed_competitions=None)
    chunk = make_chunk("a", "bcci_domestic_t20", "active", "text")
    assert matches_filter(chunk, plan) is True


def test_retrieve_bm25_no_filter_finds_all_active_bowled_chunks() -> None:
    # BM25 scores the whole corpus with no relevance threshold (shared
    # common words give even the "caught" chunk a small nonzero score), so
    # k is matched to the expected count rather than left large -- the
    # "bowled" chunks must rank strictly above it regardless.
    plan = QueryPlan(question="bowled", competition=None, allowed_competitions=None)
    results = retrieve_bm25(plan, k=3, index=index())
    ids = {sc.chunk.chunk_id for sc in results}
    assert ids == {"laws_bowled", "ipl_bowled", "icc_t20i_bowled"}


def test_retrieve_bm25_respects_competition_filter() -> None:
    # Same "bowled" query text matches all three, but only ipl+laws are
    # allowed -- the ICC T20I chunk must never appear, however it scores.
    plan = QueryPlan(question="bowled", competition="ipl", allowed_competitions=["ipl", "laws"])
    results = retrieve_bm25(plan, k=2, index=index())
    ids = {sc.chunk.chunk_id for sc in results}
    assert ids == {"laws_bowled", "ipl_bowled"}
    assert "icc_t20i_bowled" not in ids


def test_retrieve_bm25_respects_status_filter() -> None:
    plan = QueryPlan(question="caught", competition=None, allowed_competitions=None)
    results = retrieve_bm25(plan, k=10, index=index())
    ids = {sc.chunk.chunk_id for sc in results}
    assert "laws_caught" in ids
    assert "laws_caught_old" not in ids


def test_retrieve_bm25_filter_is_never_bypassed_even_at_low_k() -> None:
    # A disallowed chunk must not appear even when k is large enough that
    # it would otherwise have fit -- filtering happens before truncation,
    # not just for display.
    plan = QueryPlan(question="bowled", competition="ipl", allowed_competitions=["ipl", "laws"])
    results = retrieve_bm25(plan, k=100, index=index())
    assert all(sc.chunk.competition in {"ipl", "laws"} for sc in results)
    assert all(sc.chunk.status == "active" for sc in results)


def test_retrieve_bm25_truncates_to_k() -> None:
    plan = QueryPlan(question="batter out", competition=None, allowed_competitions=None)
    results = retrieve_bm25(plan, k=2, index=index())
    assert len(results) == 2


def test_retrieve_bm25_ranked_by_score_descending() -> None:
    plan = QueryPlan(question="bowled", competition=None, allowed_competitions=None)
    results = retrieve_bm25(plan, k=10, index=index())
    scores = [sc.score for sc in results]
    assert scores == sorted(scores, reverse=True)
