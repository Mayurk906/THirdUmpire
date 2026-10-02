"""Competition detection -> a QueryPlan with the metadata filter to apply.

Phase 4 (config/listing_pages.yaml + scraper/) added ICC's standing Test and
ODI playing conditions and BCCI's domestic playing conditions (Ranji Trophy
/ Vijay Hazare Trophy / Syed Mushtaq Ali Trophy), so their keywords and
allowed-competition entries are added here too.

Keywords are checked in two tiers, not one flat list:

1. BCCI's domestic-tournament proper nouns (Ranji/Vijay Hazare/Syed Mushtaq
   Ali Trophy) are unambiguous enough to win immediately, first-match-wins,
   even if a generic pattern below also happens to match -- e.g. "Ranji
   Trophy Test match follow-on rules" is someone describing the Ranji
   Trophy loosely as "Test match"-style cricket, not a genuine comparison.
2. The generic ICC/IPL format patterns (ipl, t20i, test match, odi) are
   checked together, and only resolve to a competition if EXACTLY ONE of
   them matches. A question naming several of these (e.g. "how does the
   over rate differ between Test, ODI and T20I?") is a real comparison
   across competitions, not a question about just one of them -- picking
   whichever pattern happened to be listed first (the original bug here)
   silently dropped the other competitions' documents from retrieval,
   producing a confident-looking "not found" for a perfectly answerable
   question instead of actually comparing them.

A question that matches nothing (or is ambiguous per rule 2) gets no
competition filter and searches every competition, which is the safe
default -- so these patterns are kept conservative (high-precision) rather
than trying to catch every possible phrasing.
"""
from __future__ import annotations

import re

from src.schemas import QueryPlan

_DOMESTIC_TROPHY_KEYWORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\branji(\s+trophy)?\b|\bdomestic\s+multi-?day\b", re.IGNORECASE), "bcci_domestic_multiday"),
    (re.compile(r"\bvijay\s+hazare(\s+trophy)?\b|\bdomestic\s+odi\b|\bdomestic\s+one-?day\b", re.IGNORECASE), "bcci_domestic_odi"),
    (re.compile(r"\bsyed\s+mushtaq\s+ali(\s+trophy)?\b|\bdomestic\s+t20\b", re.IGNORECASE), "bcci_domestic_t20"),
]

_GENERIC_FORMAT_KEYWORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(ipl|indian premier league)\b", re.IGNORECASE), "ipl"),
    (re.compile(r"\b(t20i|t20 international)\b", re.IGNORECASE), "icc_t20i"),
    (re.compile(r"\b(icc\s+)?test\s+match(es)?\b", re.IGNORECASE), "icc_test"),
    (re.compile(r"\bodi\b|\bone[- ]day international\b", re.IGNORECASE), "icc_odi"),
]

# A bare "Test" is too loose to name a competition on its own, but next to
# another format ("between Test, ODI and T20I") it can only mean Test cricket.
_LOOSE_TEST_RE = re.compile(r"\btests?\b", re.IGNORECASE)

# Rule precedence (PROJECT_PLAN.md section 1): a competition-specific
# document is checked first, falling back to the Laws where it's silent.
_ALLOWED_COMPETITIONS: dict[str, list[str]] = {
    "ipl": ["ipl", "laws"],
    "icc_t20i": ["icc_t20i", "laws"],
    "icc_test": ["icc_test", "laws"],
    "icc_odi": ["icc_odi", "laws"],
    "bcci_domestic_multiday": ["bcci_domestic_multiday", "laws"],
    "bcci_domestic_odi": ["bcci_domestic_odi", "laws"],
    "bcci_domestic_t20": ["bcci_domestic_t20", "laws"],
    "laws": ["laws"],
}


def _domestic_match(question: str) -> str | None:
    for pattern, competition in _DOMESTIC_TROPHY_KEYWORDS:
        if pattern.search(question):
            return competition
    return None


def _generic_matches(question: str) -> set[str]:
    matched = {competition for pattern, competition in _GENERIC_FORMAT_KEYWORDS if pattern.search(question)}
    if matched and _LOOSE_TEST_RE.search(question):
        matched.add("icc_test")
    return matched


def detect_competition(question: str) -> str | None:
    """See the module docstring for the two-tier rationale."""
    domestic = _domestic_match(question)
    if domestic:
        return domestic
    matched = _generic_matches(question)
    if len(matched) == 1:
        return matched.pop()
    return None


def detect_compared_competitions(question: str) -> list[str] | None:
    """The competitions a comparison question names (two or more generic
    formats), so the pipeline can make sure each one is represented in the
    answer's context -- otherwise near-identical clauses from one format
    can fill every context slot and squeeze another format out entirely."""
    if _domestic_match(question):
        return None
    matched = _generic_matches(question)
    return sorted(matched) if len(matched) >= 2 else None


def build_query_plan(question: str, competition_override: str | None = None) -> QueryPlan:
    """Build a QueryPlan. `competition_override` is the UI dropdown's explicit choice."""
    competition = competition_override or detect_competition(question)
    allowed = _ALLOWED_COMPETITIONS.get(competition) if competition else None
    compared = None if competition_override else detect_compared_competitions(question)
    return QueryPlan(
        question=question,
        competition=competition,
        allowed_competitions=allowed,
        compared_competitions=compared,
    )
