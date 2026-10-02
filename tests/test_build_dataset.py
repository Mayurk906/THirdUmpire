from pathlib import Path

from src.schemas import Chunk
from train.build_dataset import (
    build_rows,
    compute_leaked_qids,
    group_key,
    load_golden_questions,
    load_negatives,
    load_synthetic,
    split_groups,
    write_jsonl,
)
from train.gen_synthetic import SyntheticQuestion
from train.mine_negatives import MinedNegatives


def make_question(qid, pos_chunk_id="doc_a::1.1::0", doc_id="doc_a", clause="1.1", question="Q?") -> SyntheticQuestion:
    return SyntheticQuestion(qid=qid, question=question, pos_chunk_id=pos_chunk_id, doc_id=doc_id, clause=clause, competition="laws")


def make_chunk(chunk_id: str, doc_id: str, clause: str, text: str = "text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        body="mcc",
        competition="laws",
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


# --- fake embedder: returns a one-hot-ish vector based on a keyword, so
# "similar" texts can be constructed deterministically without a real model ---


def fake_embed_fn(texts: list[str]) -> list[list[float]]:
    return [[1.0, 0.0] if "ALPHA" in t else [0.0, 1.0] for t in texts]


# --- compute_leaked_qids ---


def test_leaked_qids_drops_matching_question() -> None:
    synthetic = [make_question("q1", question="ALPHA version"), make_question("q2", question="different topic")]
    leaked = compute_leaked_qids(synthetic, ["ALPHA original"], fake_embed_fn, threshold=0.85)
    assert leaked == {"q1"}


def test_leaked_qids_empty_golden_drops_nothing() -> None:
    synthetic = [make_question("q1")]
    assert compute_leaked_qids(synthetic, [], fake_embed_fn) == set()


def test_leaked_qids_empty_synthetic() -> None:
    assert compute_leaked_qids([], ["some golden question"], fake_embed_fn) == set()


def test_leaked_qids_respects_threshold() -> None:
    # A similarity of 0.0 (orthogonal fake vectors) must not be dropped at
    # a high threshold.
    synthetic = [make_question("q1", question="totally different")]
    leaked = compute_leaked_qids(synthetic, ["ALPHA golden question"], fake_embed_fn, threshold=0.85)
    assert leaked == set()


# --- group_key / split_groups ---


def test_group_key_is_doc_id_and_clause() -> None:
    sq = make_question("q1", doc_id="doc_a", clause="2.3")
    assert group_key(sq) == ("doc_a", "2.3")


def test_split_groups_respects_ratio_approximately() -> None:
    keys = [(f"doc_{i}", "1.1") for i in range(100)]
    train, val = split_groups(keys, split=0.9, seed=1)
    assert len(train) == 90
    assert len(val) == 10
    assert train.isdisjoint(val)


def test_split_groups_is_deterministic() -> None:
    keys = [(f"doc_{i}", "1.1") for i in range(50)]
    a = split_groups(keys, split=0.9, seed=7)
    b = split_groups(keys, split=0.9, seed=7)
    assert a == b


def test_split_groups_dedupes_repeated_keys() -> None:
    keys = [("doc_a", "1.1")] * 10 + [("doc_b", "1.1")]
    train, val = split_groups(keys, split=0.9, seed=1)
    assert len(train) + len(val) == 2


# --- build_rows ---


def test_build_rows_positive_label_is_one() -> None:
    sq = make_question("q1", pos_chunk_id="c1", question="Q?")
    chunks = {"c1": make_chunk("c1", "doc_a", "1.1", text="positive text")}
    rows = build_rows(sq, [], chunks)
    assert rows == [{"query": "Q?", "passage": "positive text", "label": 1}]


def test_build_rows_negatives_label_zero() -> None:
    sq = make_question("q1", pos_chunk_id="c1", question="Q?")
    chunks = {
        "c1": make_chunk("c1", "doc_a", "1.1", text="pos"),
        "c2": make_chunk("c2", "doc_a", "2.1", text="neg1"),
        "c3": make_chunk("c3", "doc_a", "3.1", text="neg2"),
    }
    rows = build_rows(sq, ["c2", "c3"], chunks)
    assert len(rows) == 3
    assert rows[0]["label"] == 1
    assert rows[1] == {"query": "Q?", "passage": "neg1", "label": 0}
    assert rows[2] == {"query": "Q?", "passage": "neg2", "label": 0}


def test_build_rows_missing_positive_returns_empty() -> None:
    sq = make_question("q1", pos_chunk_id="missing")
    rows = build_rows(sq, [], {})
    assert rows == []


def test_build_rows_skips_missing_negative_chunk() -> None:
    sq = make_question("q1", pos_chunk_id="c1")
    chunks = {"c1": make_chunk("c1", "doc_a", "1.1", text="pos")}
    rows = build_rows(sq, ["missing_neg"], chunks)
    assert len(rows) == 1  # only the positive


# --- file I/O helpers ---


def test_load_golden_questions_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_golden_questions(tmp_path / "golden.jsonl") == []


def test_load_golden_questions_extracts_question_field(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text('{"id": "g1", "question": "What is out?", "other": "x"}\n', encoding="utf-8")
    assert load_golden_questions(path) == ["What is out?"]


def test_load_negatives_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    assert load_negatives(tmp_path / "negatives.jsonl") == {}


def test_load_negatives_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "negatives.jsonl"
    write_jsonl([MinedNegatives(qid="q1", negative_chunk_ids=["a", "b"]).model_dump()], path)
    assert load_negatives(path) == {"q1": ["a", "b"]}


def test_load_synthetic_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.jsonl"
    sq = make_question("q1")
    write_jsonl([sq.model_dump()], path)
    loaded = load_synthetic(path)
    assert len(loaded) == 1
    assert loaded[0].qid == "q1"


def test_write_jsonl_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "out.jsonl"
    write_jsonl([{"a": 1}], path)
    assert path.exists()
