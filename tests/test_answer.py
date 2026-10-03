import pytest
from pydantic import ValidationError

from src import answer as answer_module
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


class _FailingStructuredLLM:
    def invoke(self, messages):
        # Deliberately no "429"/"500"/etc substring -- a retryable-looking
        # message would trigger _invoke_with_backoff's real sleep-and-retry
        # loop and make these tests slow for no reason; this is testing the
        # outer fallback-to-Groq path, not the backoff logic itself.
        raise RuntimeError("model produced malformed output")


class _FailingLLM:
    def with_structured_output(self, schema):
        return _FailingStructuredLLM()


def test_groq_fallback_used_when_gemini_fails_and_key_is_set(monkeypatch) -> None:
    monkeypatch.setattr(answer_module, "GROQ_API_KEY", "fake-key")
    groq_answer = Answer(answer="From Groq.", found=True, applies_to="laws", citations=[])
    monkeypatch.setattr(answer_module, "_generate_answer_groq", lambda question, chunks: groq_answer)

    result = generate_answer("Some question", [], llm=_FailingLLM())

    assert result == groq_answer


def test_no_groq_fallback_when_key_unset(monkeypatch) -> None:
    monkeypatch.setattr(answer_module, "GROQ_API_KEY", None)

    def _boom(question, chunks):
        raise AssertionError("Groq must not be called when GROQ_API_KEY is unset")

    monkeypatch.setattr(answer_module, "_generate_answer_groq", _boom)

    result = generate_answer("Some question", [], llm=_FailingLLM())

    assert result.found is False
    assert "invalid output" in result.answer


def test_groq_fallback_failure_returns_clean_error(monkeypatch) -> None:
    monkeypatch.setattr(answer_module, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(
        answer_module,
        "_generate_answer_groq",
        lambda question, chunks: (_ for _ in ()).throw(RuntimeError("Groq is also down")),
    )

    result = generate_answer("Some question", [], llm=_FailingLLM())

    assert result.found is False


class _FakeGroqClient:
    """Mimics groq.Groq just enough for _generate_answer_groq's happy path."""

    def __init__(self, content: str):
        message = type("Msg", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        response = type("Response", (), {"choices": [choice]})()
        self.chat = type("Chat", (), {"completions": type("Completions", (), {"create": lambda self, **kw: response})()})()


def test_generate_answer_groq_parses_json_response(monkeypatch) -> None:
    content = '{"answer": "Eleven players.", "found": true, "applies_to": "laws", "citations": []}'
    monkeypatch.setattr(answer_module, "_get_groq_client", lambda: _FakeGroqClient(content))

    result = answer_module._generate_answer_groq("How many players?", [])

    assert result.answer == "Eleven players."
    assert result.found is True
    assert result.applies_to == "laws"
