from src import config, tracing


def test_tracing_enabled_matches_config_flags() -> None:
    # Whether tracing is actually on depends on the environment's .env, not
    # on this test -- what matters is that it's derived correctly.
    expected = bool(config.LANGSMITH_TRACING and config.LANGSMITH_API_KEY)
    assert tracing.is_tracing_enabled() == expected


def test_traceable_bare_decorator_is_noop() -> None:
    @tracing.traceable
    def add(a: int, b: int) -> int:
        return a + b

    assert add(1, 2) == 3


def test_traceable_with_kwargs_is_noop() -> None:
    @tracing.traceable(name="add")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(1, 2) == 3
