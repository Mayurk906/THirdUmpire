from src.citations import validate_citations
from src.schemas import Answer, Chunk, Citation


def make_chunk(doc_id: str, clause: str, page_start: int, page_end: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::{clause}::0",
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
        page_start=page_start,
        page_end=page_end,
        chunk_index=0,
        content_hash="hash",
        status="active",
        text="some clause text",
    )


CONTEXT = [make_chunk("mcc_laws_2026", "32.1", 52, 52), make_chunk("mcc_laws_2026", "32.2", 53, 54)]


def test_valid_citation_within_page_range() -> None:
    answer = Answer(
        answer="Bowled applies.",
        found=True,
        applies_to="laws",
        citations=[Citation(doc_id="mcc_laws_2026", clause="32.1", page=52)],
    )
    valid, invalid = validate_citations(answer, CONTEXT)
    assert valid is True
    assert invalid == []


def test_citation_with_wrong_clause_is_invalid() -> None:
    answer = Answer(
        answer="x",
        found=True,
        applies_to="laws",
        citations=[Citation(doc_id="mcc_laws_2026", clause="99.9", page=52)],
    )
    valid, invalid = validate_citations(answer, CONTEXT)
    assert valid is False
    assert len(invalid) == 1


def test_citation_with_page_out_of_range_is_invalid() -> None:
    answer = Answer(
        answer="x",
        found=True,
        applies_to="laws",
        citations=[Citation(doc_id="mcc_laws_2026", clause="32.2", page=99)],
    )
    valid, invalid = validate_citations(answer, CONTEXT)
    assert valid is False
    assert invalid[0].page == 99


def test_found_true_with_no_citations_is_invalid() -> None:
    answer = Answer(answer="x", found=True, applies_to="laws", citations=[])
    valid, invalid = validate_citations(answer, CONTEXT)
    assert valid is False
    assert invalid == []  # nothing malformed, just missing


def test_found_false_with_no_citations_is_valid() -> None:
    answer = Answer(answer="Not found in the provided rule documents.", found=False, applies_to="unknown", citations=[])
    valid, invalid = validate_citations(answer, CONTEXT)
    assert valid is True


def test_subclause_citation_is_valid_against_parent_chunk() -> None:
    # 32.1.5 is a sub-clause folded into the 32.1 chunk (see chunk.py); the
    # model may cite the more specific number it actually found in the text.
    answer = Answer(
        answer="x",
        found=True,
        applies_to="laws",
        citations=[Citation(doc_id="mcc_laws_2026", clause="32.1.5", page=52)],
    )
    valid, invalid = validate_citations(answer, CONTEXT)
    assert valid is True
    assert invalid == []


def test_subclause_citation_does_not_falsely_match_similar_prefix() -> None:
    # "32.15" must not match the "32.1" chunk just because it starts with
    # the same characters -- only a real dotted sub-clause should match.
    context = [make_chunk("mcc_laws_2026", "32.1", 52, 52)]
    answer = Answer(
        answer="x",
        found=True,
        applies_to="laws",
        citations=[Citation(doc_id="mcc_laws_2026", clause="32.15", page=52)],
    )
    valid, invalid = validate_citations(answer, context)
    assert valid is False
    assert len(invalid) == 1


def test_invalid_citations_are_kept_not_removed() -> None:
    answer = Answer(
        answer="x",
        found=True,
        applies_to="laws",
        citations=[
            Citation(doc_id="mcc_laws_2026", clause="32.1", page=52),
            Citation(doc_id="mcc_laws_2026", clause="99.9", page=1),
        ],
    )
    valid, invalid = validate_citations(answer, CONTEXT)
    assert valid is False
    assert len(answer.citations) == 2  # untouched
    assert len(invalid) == 1
