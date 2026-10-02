"""LangSmith tracing setup.

Tracing is enabled only when LANGSMITH_TRACING=true and LANGSMITH_API_KEY is
set (see src/config.py); otherwise every helper here is a no-op, and the
project must run correctly with tracing disabled.

LangChain calls (Gemini) are traced automatically once LANGSMITH_TRACING /
LANGSMITH_API_KEY / LANGSMITH_PROJECT are in the environment (python-dotenv
already loads .env into os.environ via src.config). Non-LangChain steps
(query analysis, retrieval, fusion, reranking, citation validation) should be
wrapped with the @traceable decorator exported from this module.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

from src import config

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

TRACING_ENABLED = bool(config.LANGSMITH_TRACING and config.LANGSMITH_API_KEY)

try:
    from langsmith import traceable as _langsmith_traceable
except ImportError:
    _langsmith_traceable = None

if config.LANGSMITH_TRACING and not config.LANGSMITH_API_KEY:
    logger.warning("LANGSMITH_TRACING is true but LANGSMITH_API_KEY is unset; tracing is disabled.")

if TRACING_ENABLED and _langsmith_traceable is None:
    logger.warning("LangSmith tracing is enabled but the 'langsmith' package isn't installed; tracing is disabled.")
    TRACING_ENABLED = False


def is_tracing_enabled() -> bool:
    return TRACING_ENABLED


def _noop_decorator(*dargs: Any, **dkwargs: Any) -> Any:
    # Support both bare `@traceable` and `@traceable(name=..., metadata=...)`.
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        func = dargs[0]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def traceable(*dargs: Any, **dkwargs: Any) -> Any:
    """Drop-in replacement for langsmith.traceable that no-ops when tracing is disabled."""
    if TRACING_ENABLED and _langsmith_traceable is not None:
        return _langsmith_traceable(*dargs, **dkwargs)
    return _noop_decorator(*dargs, **dkwargs)
