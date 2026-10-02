from pathlib import Path

from src.schemas import Chunk, ScoredChunk
from train.gen_synthetic import SyntheticQuestion
from train.mine_negatives import (
    MinedNegatives,
    append_result,
    filter_candidates,
    is_positive,
    load_done_qids,
)


def make_chunk(chunk_id: str, doc_id: str, clause: str, competition: str = "laws", text: str = "text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        body="mcc",
        competition=competition,
        gender="all",
        doc_type="laws",
        effective_from="2026-01-01",
        law_no=1,
        clause=clause,
        clause_title="Title",
        section_path=f"LAW 1 > {clause}",
        page_start=1,
        page_end=1,
        chunk_index=0,
        content_hash="hash",
        status="active",
        text=text,
    )


def scored(chunk: Chunk, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score)


def make_question(qid="q1", pos_chunk_id="doc_a::1.1::0", doc_id="doc_a", clause="1.1", competition="laws") -> SyntheticQuestion:
    return SyntheticQuestion(qid=qid, question="Q?", pos_chunk_id=pos_chunk_id, doc_id=doc_id, clause=clause, competition=competition)


def fixed_similarity(value: float):
    return lambda a, b: value


# --- is_positive ---


def test_is_positive_matches_doc_id_and_clause() -> None:
    sq = make_question(doc_id="doc_a", clause="1.1")
    chunk = scored(make_chunk("doc_a::1.1::0", "doc_a", "1.1"))
    assert is_positive(chunk, sq) is True


def test_is_positive_false_for_different_clause_same_doc() -> None:
    sq = make_question(doc_id="doc_a", clause="1.1")
    chunk = scored(make_chunk("doc_a::1.2::0", "doc_a", "1.2"))
    assert is_positive(chunk, sq) is False


def test_is_positive_false_for_same_clause_different_doc() -> None:
    # Clause numbers repeat across documents (e.g. "40.1" in every playing
    # conditions doc) -- doc_id must be checked too, not just clause.
    sq = make_question(doc_id="doc_a", clause="40.1")
    chunk = scored(make_chunk("doc_b::40.1::0", "doc_b", "40.1"))
    assert is_positive(chunk, sq) is False


# --- filter_candidates ---


def test_filter_candidates_excludes_the_positive_itself() -> None:
    sq = make_question(doc_id="doc_a", clause="1.1")
    retrieved = [scored(make_chunk("doc_a::1.1::0", "doc_a", "1.1")), scored(make_chunk("doc_a::1.2::0", "doc_a", "1.2"))]
    kept = filter_candidates(sq, retrieved, positive_text="pos", similarity_fn=fixed_similarity(0.0))
    assert [c.chunk.chunk_id for c in kept] == ["doc_a::1.2::0"]


def test_filter_candidates_no_competition_drops_near_duplicate_text() -> None:
    # No competition named -> the false-negative similarity guard applies.
    sq = make_question(competition="laws", doc_id="doc_a", clause="1.1")
    retrieved = [scored(make_chunk("doc_b::2.1::0", "doc_b", "2.1"))]
    kept = filter_candidates(sq, retrieved, positive_text="pos", similarity_fn=fixed_similarity(0.99))
    assert kept == []


def test_filter_candidates_no_competition_keeps_dissimilar_text() -> None:
    sq = make_question(competition="laws", doc_id="doc_a", clause="1.1")
    retrieved = [scored(make_chunk("doc_b::2.1::0", "doc_b", "2.1"))]
    kept = filter_candidates(sq, retrieved, positive_text="pos", similarity_fn=fixed_similarity(0.5))
    assert len(kept) == 1


def test_filter_candidates_with_competition_keeps_similar_same_clause_other_competition() -> None:
    # Question names a competition -> same-clause chunks from OTHER
    # competitions are kept as intended hard negatives even if the text is
    # near-identical (their context header differs, which is the point).
    sq = make_question(competition="ipl", doc_id="ipl_2026_pc", clause="40.1")
    retrieved = [scored(make_chunk("icc_mens_t20i_2025_07::40.1::0", "icc_mens_t20i_2025_07", "40.1", competition="icc_t20i"))]
    kept = filter_candidates(sq, retrieved, positive_text="pos", similarity_fn=fixed_similarity(0.99))
    assert len(kept) == 1


def test_filter_candidates_caps_at_max_negatives() -> None:
    sq = make_question(doc_id="doc_a", clause="1.1")
    retrieved = [scored(make_chunk(f"doc_a::{i}.1::0", "doc_a", f"{i}.1")) for i in range(2, 10)]
    kept = filter_candidates(sq, retrieved, positive_text="pos", similarity_fn=fixed_similarity(0.0))
    assert len(kept) == 4  # MAX_NEGATIVES_PER_POSITIVE


def test_filter_candidates_prefers_highest_ranked() -> None:
    # retrieved is assumed already sorted by relevance descending; the cap
    # must keep the first N (highest ranked), not an arbitrary subset.
    sq = make_question(doc_id="doc_a", clause="1.1")
    retrieved = [scored(make_chunk(f"doc_a::{i}.1::0", "doc_a", f"{i}.1"), score=1.0 - i * 0.01) for i in range(2, 10)]
    kept = filter_candidates(sq, retrieved, positive_text="pos", similarity_fn=fixed_similarity(0.0))
    assert [c.chunk.chunk_id for c in kept] == ["doc_a::2.1::0", "doc_a::3.1::0", "doc_a::4.1::0", "doc_a::5.1::0"]


# --- load_done_qids / append_result (file I/O) ---


def test_load_done_qids_missing_file(tmp_path: Path) -> None:
    assert load_done_qids(tmp_path / "negatives.jsonl") == set()


def test_append_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "negatives.jsonl"
    append_result(MinedNegatives(qid="q1", negative_chunk_ids=["a", "b"]), path)
    assert load_done_qids(path) == {"q1"}


def test_append_result_is_additive(tmp_path: Path) -> None:
    path = tmp_path / "negatives.jsonl"
    append_result(MinedNegatives(qid="q1", negative_chunk_ids=[]), path)
    append_result(MinedNegatives(qid="q2", negative_chunk_ids=[]), path)
    assert load_done_qids(path) == {"q1", "q2"}
