import pytest

from src import config


def test_defaults() -> None:
    assert config.EMBED_MODEL == "BAAI/bge-small-en-v1.5"
    assert config.RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.RETRIEVE_K == 20
    assert config.CONTEXT_K == 5
    assert config.MAX_CHUNK_CHARS == 1500
    assert config.CHUNK_OVERLAP == 150
    assert config.MAX_QUESTION_CHARS == 500
    assert config.PROMPT_VERSION == "v3"
    assert config.TEMPERATURE == 0
    assert config.CHROMA_DIR == "chroma_db"
    assert config.COLLECTION == "cricket_rules"
    assert config.RRF_K == 60
    assert config.DEFAULT_FIRST_STAGE == "dense"
    assert config.SYNTHETIC_QUESTIONS_MIN == 2
    assert config.SYNTHETIC_QUESTIONS_MAX == 3
    assert config.SYNTHETIC_CHUNKS_PER_REQUEST == 25
    assert config.SYNTHETIC_SUBSET_SIZE == 600
    assert config.LEAKAGE_SIMILARITY_THRESHOLD == 0.85
    assert config.NEGATIVE_MINING_TOP_K == 30
    assert config.MAX_NEGATIVES_PER_POSITIVE == 4
    assert config.FALSE_NEGATIVE_SIMILARITY_THRESHOLD == 0.97
    assert config.TRAIN_VAL_SPLIT == 0.9
    assert config.RERANKER_FT_EPOCHS == 2
    assert config.RERANKER_FT_BATCH_SIZE == 16


def test_require_gemini_model_raises_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "GEMINI_MODEL", None)
    with pytest.raises(RuntimeError):
        config.require_gemini_model()


def test_require_gemini_model_returns_value_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "GEMINI_MODEL", "gemini-2.5-flash")
    assert config.require_gemini_model() == "gemini-2.5-flash"


def test_require_synthetic_gemini_model_uses_separate_model_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.setattr(config, "SYNTHETIC_GEMINI_MODEL", "gemini-flash-lite-latest")
    assert config.require_synthetic_gemini_model() == "gemini-flash-lite-latest"


def test_require_synthetic_gemini_model_falls_back_to_gemini_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.setattr(config, "SYNTHETIC_GEMINI_MODEL", None)
    assert config.require_synthetic_gemini_model() == "gemini-3.6-flash"


def test_require_synthetic_gemini_model_raises_when_both_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "GEMINI_MODEL", None)
    monkeypatch.setattr(config, "SYNTHETIC_GEMINI_MODEL", None)
    with pytest.raises(RuntimeError):
        config.require_synthetic_gemini_model()
