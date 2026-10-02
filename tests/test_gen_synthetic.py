from pathlib import Path

from src.schemas import Chunk
from train.gen_synthetic import (
    SyntheticQuestion,
    _BatchResponse,
    _ChunkQuestions,
    append_results,
    build_batches,
    build_prompt,
    competition_label,
    load_done_chunk_ids,
    parse_batch_response,
    select_subset,
)


def make_chunk(chunk_id: str, doc_id: str, competition: str = "laws", gender: str = "all") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        body="mcc",
        competition=competition,
        gender=gender,
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
        status="active",
        text="Some clause text.",
    )


# --- competition_label ---


def test_competition_label_laws_is_none() -> None:
    assert competition_label("laws") is None


def test_competition_label_ipl() -> None:
    assert competition_label("ipl") == "IPL"


def test_competition_label_bcci_domestic_mens_names_the_tournament() -> None:
    assert "Ranji Trophy" in competition_label("bcci_domestic_multiday", gender="men")
    assert "Vijay Hazare Trophy" in competition_label("bcci_domestic_odi", gender="men")
    assert "Syed Mushtaq Ali Trophy" in competition_label("bcci_domestic_t20", gender="men")


def test_competition_label_bcci_domestic_defaults_to_mens_tournament_name() -> None:
    # gender="all" (the default) must not silently fall back to the
    # women's-safe generic phrasing -- only an explicit gender="women"
    # should.
    assert "Ranji Trophy" in competition_label("bcci_domestic_multiday")


def test_competition_label_bcci_domestic_womens_does_not_use_mens_trophy_name() -> None:
    # Regression test: a fine-tuned reranker trained when this used the same
    # men's-only trophy names for women's chunks too measurably confused
    # men's/women's BCCI documents on the real eval set (see git history /
    # PROJECT_PLAN.md Phase 6 fine-tuning notes).
    label = competition_label("bcci_domestic_multiday", gender="women")
    assert "Ranji Trophy" not in label
    assert "women" in label.lower()


def test_competition_label_icc_competitions_are_gender_neutral_phrasing() -> None:
    # "a T20I" is correct regardless of gender -- there's no separate proper
    # noun to get wrong here, unlike the BCCI domestic trophies.
    assert competition_label("icc_t20i", gender="men") == competition_label("icc_t20i", gender="women")


# --- select_subset ---


def test_select_subset_returns_all_when_size_exceeds_corpus() -> None:
    chunks = [make_chunk(f"c{i}", "doc_a") for i in range(3)]
    assert len(select_subset(chunks, 10)) == 3


def test_select_subset_stratifies_across_doc_ids() -> None:
    chunks = [make_chunk(f"a{i}", "doc_a") for i in range(10)] + [make_chunk(f"b{i}", "doc_b") for i in range(2)]
    subset = select_subset(chunks, 4, seed=1)
    doc_ids = {c.doc_id for c in subset}
    # Both docs should be represented, not just doc_a (which has more chunks).
    assert doc_ids == {"doc_a", "doc_b"}
    assert len(subset) == 4


def test_select_subset_is_deterministic_for_a_given_seed() -> None:
    chunks = [make_chunk(f"c{i}", "doc_a") for i in range(20)]
    a = select_subset(chunks, 5, seed=7)
    b = select_subset(chunks, 5, seed=7)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_select_subset_no_duplicates() -> None:
    chunks = [make_chunk(f"c{i}", "doc_a") for i in range(20)]
    subset = select_subset(chunks, 15, seed=3)
    ids = [c.chunk_id for c in subset]
    assert len(ids) == len(set(ids))


# --- build_batches ---


def test_build_batches_splits_into_groups() -> None:
    chunks = [make_chunk(f"c{i}", "doc_a") for i in range(7)]
    batches = build_batches(chunks, batch_size=3)
    assert [len(b) for b in batches] == [3, 3, 1]


def test_build_batches_empty_input() -> None:
    assert build_batches([], batch_size=5) == []


# --- build_prompt ---


def test_build_prompt_names_competition_for_specific_chunk() -> None:
    chunk = make_chunk("c1", "doc_a", competition="ipl")
    prompt = build_prompt([chunk])
    assert "specific to IPL" in prompt


def test_build_prompt_general_note_for_laws_chunk() -> None:
    chunk = make_chunk("c1", "doc_a", competition="laws")
    prompt = build_prompt([chunk])
    assert "general (Laws of Cricket)" in prompt


def test_build_prompt_includes_chunk_text() -> None:
    chunk = make_chunk("c1", "doc_a")
    prompt = build_prompt([chunk])
    assert "Some clause text." in prompt


def test_build_prompt_uses_mens_trophy_name_for_mens_bcci_chunk() -> None:
    chunk = make_chunk("c1", "bcci_mens_domestic_multiday_2025", competition="bcci_domestic_multiday", gender="men")
    prompt = build_prompt([chunk])
    assert "Ranji Trophy" in prompt


def test_build_prompt_does_not_use_mens_trophy_name_for_womens_bcci_chunk() -> None:
    chunk = make_chunk("c1", "bcci_womens_domestic_multiday_2025", competition="bcci_domestic_multiday", gender="women")
    prompt = build_prompt([chunk])
    chunk_note = prompt.split("[Chunk 0]")[1]
    assert "Ranji Trophy" not in chunk_note
    assert "women" in chunk_note.lower()


# --- parse_batch_response ---


def test_parse_batch_response_generates_qid_per_question() -> None:
    chunks = [make_chunk("doc_a::1.1::0", "doc_a")]
    response = _BatchResponse(items=[_ChunkQuestions(chunk_index=0, questions=["Q1?", "Q2?"])])
    results = parse_batch_response(chunks, response)
    assert [r.qid for r in results] == ["syn_doc_a::1.1::0_0", "syn_doc_a::1.1::0_1"]
    assert all(r.pos_chunk_id == "doc_a::1.1::0" for r in results)


def test_parse_batch_response_missing_chunk_index_yields_no_questions() -> None:
    chunks = [make_chunk("c1", "doc_a"), make_chunk("c2", "doc_a")]
    response = _BatchResponse(items=[_ChunkQuestions(chunk_index=0, questions=["Q1?"])])
    results = parse_batch_response(chunks, response)
    assert len(results) == 1
    assert results[0].pos_chunk_id == "c1"


def test_parse_batch_response_carries_doc_id_clause_competition() -> None:
    chunk = make_chunk("c1", "doc_a", competition="icc_t20i")
    response = _BatchResponse(items=[_ChunkQuestions(chunk_index=0, questions=["Q?"])])
    results = parse_batch_response([chunk], response)
    assert results[0].doc_id == "doc_a"
    assert results[0].clause == "1.1"
    assert results[0].competition == "icc_t20i"


# --- load_done_chunk_ids / append_results (file I/O) ---


def test_load_done_chunk_ids_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_done_chunk_ids(tmp_path / "synthetic.jsonl") == set()


def test_append_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.jsonl"
    results = [
        SyntheticQuestion(qid="q1", question="Q?", pos_chunk_id="c1", doc_id="doc_a", clause="1.1", competition="laws")
    ]
    append_results(results, path)
    assert load_done_chunk_ids(path) == {"c1"}


def test_append_results_is_additive(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.jsonl"
    r1 = [SyntheticQuestion(qid="q1", question="Q1?", pos_chunk_id="c1", doc_id="doc_a", clause="1.1", competition="laws")]
    r2 = [SyntheticQuestion(qid="q2", question="Q2?", pos_chunk_id="c2", doc_id="doc_a", clause="1.2", competition="laws")]
    append_results(r1, path)
    append_results(r2, path)
    assert load_done_chunk_ids(path) == {"c1", "c2"}
