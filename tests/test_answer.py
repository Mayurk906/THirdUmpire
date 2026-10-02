import pytest
from pydantic import ValidationError

from src.answer import QuestionTooLongError, _format_context, generate_answer
from src.config import MAX_QUESTION_CHARS
from src.schemas import Answer, Chunk


class _NeverCalledLLM:
    """Fails the test if the LLM is ever touched."""

    def with_structured_output(self, schema):
        raise AssertionError("the LLM must not be called for an over-length question")


class _FakeStructuredLLM:
    def __init__(self, response):
        self._response = response

    def invoke(self, messages):
        return self._response


class _FakeLLM:
    def __init__(self, response):
        self._structured = _FakeStructuredLLM(response)

    def with_structured_output(self, schema):
        return self._structured


def test_over_length_question_rejected_without_calling_llm() -> None:
    long_question = "x" * (MAX_QUESTION_CHARS + 1)
    with pytest.raises(QuestionTooLongError):
        generate_answer(long_question, [], llm=_NeverCalledLLM())


def test_invalid_applies_to_value_fails_validation() -> None:
    with pytest.raises(ValidationError):
        Answer(answer="x", found=True, applies_to="not_a_real_competition", citations=[])


def test_not_found_path_with_empty_context() -> None:
    expected = Answer(
        answer="Not found in the provided rule documents.",
        found=False,
        applies_to="unknown",
        citations=[],
    )
    result = generate_answer("Some question with no matching context", [], llm=_FakeLLM(expected))
    assert result == expected


def test_generate_answer_returns_llm_result_directly() -> None:
    expected = Answer(answer="Yes, bowled applies.", found=True, applies_to="laws", citations=[])
    result = generate_answer("Is it out bowled?", [], llm=_FakeLLM(expected))
    assert result == expected


def _make_chunk(doc_id: str, clause: str, page: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::{clause}::0",
        doc_id=doc_id,
        body="bcci",
        competition="bcci_domestic_multiday",
        gender="men",
        doc_type="playing_conditions",
        effective_from="2025",
        law_no=-1,
        clause=clause,
        clause_title="Runners",
        section_path=f"25 {clause}",
        page_start=page,
        page_end=page,
        chunk_index=0,
        content_hash="hash",
        status="active",
        text="Runners shall not be permitted.",
    )


def test_format_context_label_includes_doc_id_not_just_title() -> None:
    # Regression test: the model can only cite what it can see. The label
    # must expose the real doc_id explicitly -- a title-only label (as this
    # used to be) leads the model to cite the title string as the doc_id,
    # which then always fails citation validation against the chunk's
    # actual doc_id (see git history for the bug this caught live).
    chunk = _make_chunk("bcci_mens_domestic_multiday_2025", "25.5", 59)
    context = _format_context([chunk], {"bcci_mens_domestic_multiday_2025": "Playing Conditions for Men's Multi-day Matches"})
    assert "doc_id=bcci_mens_domestic_multiday_2025" in context
    assert "Playing Conditions for Men's Multi-day Matches" in context
    assert "25.5" in context
    assert "p.59" in context


def test_format_context_falls_back_to_doc_id_when_title_unknown() -> None:
    chunk = _make_chunk("unknown_doc", "1.1", 1)
    context = _format_context([chunk], {})
    assert "doc_id=unknown_doc" in context
