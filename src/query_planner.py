"""Optional LLM-based query planning -- an alternative to query_analyzer.py's
regex detector. See config.py's QUERY_PLANNER_MODE for why it's off by
default and what it falls back to.

The regex detector is fast, free, and good at the patterns it was written
for, but it has two real gaps this covers:
  - A bare "Test" only counts as a format when ANOTHER format pattern also
    matches in the same question (see query_analyzer.py's _LOOSE_TEST_RE
    comment) -- a question that's unambiguous to a human ("What's the over
    rate in a Test?") but names only one format gets no competition filter.
  - Gender is never used as a filter at all -- "What's the powerplay in
    women's T20Is?" and the men's equivalent currently retrieve and rank
    identically; only select_context()'s regex-based preference among
    already-retrieved chunks sees "women's", and only via the raw question
    text, not a planner's understanding of it.
An LLM reads both correctly without hand-written patterns for every phrasing.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field

from src.config import TEMPERATURE, require_query_planner_gemini_model
from src.query_analyzer import _ALLOWED_COMPETITIONS, build_query_plan
from src.schemas import KNOWN_COMPETITIONS, QueryPlan

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = f"""You read a cricket rules question and identify which \
competition(s) it's about, for a retrieval system to filter on.

Known competitions: {", ".join(KNOWN_COMPETITIONS)}.

Rules:
- competitions: every competition the question is actually about, usually \
one. If the question compares formats (e.g. "Test vs ODI", "how does the \
over rate differ between Test, ODI and T20I"), list every one of them -- a \
bare "Test" counts if another format is named alongside it, even informally.
- If no specific competition is named or implied, return an empty list \
(every competition and the Laws of Cricket will be searched).
- gender: "women" only if the question explicitly says "women's" (or \
equivalent, e.g. "ladies'"), "men" only if it explicitly says "men's", \
otherwise "unspecified" -- never guess gender from context.
"""


class _CompetitionList(BaseModel):
    competitions: list[
        Literal[
            "laws",
            "ipl",
            "icc_t20i",
            "icc_test",
            "icc_odi",
            "bcci_domestic_multiday",
            "bcci_domestic_odi",
            "bcci_domestic_t20",
        ]
    ] = Field(default_factory=list)
    gender: Literal["men", "women", "unspecified"] = "unspecified"


@lru_cache(maxsize=1)
def _get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=require_query_planner_gemini_model(), temperature=TEMPERATURE)


def _plan_with_llm(question: str) -> _CompetitionList | None:
    """Returns None on ANY failure (bad structured output, network error, an
    exhausted free-tier quota) so the caller can fall back to the regex
    detector -- a planner failure must never be worse than having no
    planner at all."""
    try:
        structured_llm = _get_llm().with_structured_output(_CompetitionList)
        result = structured_llm.invoke([("system", _SYSTEM_PROMPT), ("human", question)])
        return result if isinstance(result, _CompetitionList) else _CompetitionList.model_validate(result)
    except Exception:
        logger.warning("Query planner LLM call failed; falling back to the regex detector.", exc_info=True)
        return None


def build_query_plan_llm(question: str, competition_override: str | None = None) -> QueryPlan:
    """LLM-first version of query_analyzer.build_query_plan(). Falls back to
    the regex detector's plan whenever the LLM call fails, or when
    `competition_override` is set (an explicit override needs no planning)."""
    if competition_override:
        return build_query_plan(question, competition_override)

    llm_plan = _plan_with_llm(question)
    if llm_plan is None:
        return build_query_plan(question)

    gender = llm_plan.gender if llm_plan.gender != "unspecified" else None
    competitions = llm_plan.competitions

    if len(competitions) >= 2:
        return QueryPlan(
            question=question,
            competition=None,
            allowed_competitions=None,  # top_up_compared() fetches each one specifically
            compared_competitions=sorted(competitions),
            gender=gender,
        )
    if len(competitions) == 1:
        competition = competitions[0]
        return QueryPlan(
            question=question,
            competition=competition,
            allowed_competitions=_ALLOWED_COMPETITIONS.get(competition),
            compared_competitions=None,
            gender=gender,
        )
    return QueryPlan(question=question, competition=None, allowed_competitions=None, compared_competitions=None, gender=gender)
