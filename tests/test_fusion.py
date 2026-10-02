from src.fusion import reciprocal_rank_fusion
from src.schemas import Chunk, ScoredChunk


def make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        body="mcc",
        competition="laws",
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
        status="active",
        text="text",
    )


def scored(chunk_id: str, score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(chunk=make_chunk(chunk_id), score=score)


def test_rrf_chunk_in_both_lists_ranks_above_either_alone() -> None:
    # "a" is rank 1 in both lists; "b" is rank 1 in only the first.
    dense = [scored("a"), scored("b")]
    bm25 = [scored("a"), scored("c")]
    fused = reciprocal_rank_fusion([dense, bm25], k=10, rrf_k=60)
    assert [sc.chunk.chunk_id for sc in fused][0] == "a"


def test_rrf_known_scores_on_toy_lists() -> None:
    # a: rank 1 in list1 (1/61), rank 2 in list2 (1/62)
    # b: rank 2 in list1 (1/62), absent from list2
    # c: absent from list1, rank 1 in list2 (1/61)
    list1 = [scored("a"), scored("b")]
    list2 = [scored("c"), scored("a")]
    fused = reciprocal_rank_fusion([list1, list2], k=10, rrf_k=60)
    scores = {sc.chunk.chunk_id: sc.score for sc in fused}

    assert scores["a"] == 1 / 61 + 1 / 62
    assert scores["b"] == 1 / 62
    assert scores["c"] == 1 / 61
    # Known expected order: a (in both) > c > b (1/61 > 1/62).
    assert [sc.chunk.chunk_id for sc in fused] == ["a", "c", "b"]


def test_rrf_truncates_to_k() -> None:
    dense = [scored("a"), scored("b"), scored("c")]
    fused = reciprocal_rank_fusion([dense], k=2, rrf_k=60)
    assert len(fused) == 2
    assert [sc.chunk.chunk_id for sc in fused] == ["a", "b"]


def test_rrf_single_list_preserves_relative_order() -> None:
    dense = [scored("x"), scored("y"), scored("z")]
    fused = reciprocal_rank_fusion([dense], k=10)
    assert [sc.chunk.chunk_id for sc in fused] == ["x", "y", "z"]


def test_rrf_empty_lists() -> None:
    assert reciprocal_rank_fusion([[], []], k=10) == []


def test_rrf_merges_by_chunk_id_not_object_identity() -> None:
    # The same logical chunk_id appears as two distinct ScoredChunk objects
    # (as it would from two independent retrieval stages).
    dense = [scored("a", score=0.9)]
    bm25 = [scored("a", score=5.3)]
    fused = reciprocal_rank_fusion([dense, bm25], k=10, rrf_k=60)
    assert len(fused) == 1
    assert fused[0].score == 1 / 61 + 1 / 61
