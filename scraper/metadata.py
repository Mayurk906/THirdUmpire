"""Parse (gender, format, effective date) out of a listing page's link title.

Pure, dependency-free and unit-tested against real titles observed on the
ICC and BCCI documents pages (PROJECT_PLAN.md Phase 4). Deciding what
`competition`/`doc_type` a title maps to is business logic that varies by
listing page, so it stays in scraper/scrape.py; this module only extracts
the raw signals out of the title text itself.
"""
from __future__ import annotations

import re

_MONTHS = {
    "jan": "01", "january": "01",
    "feb": "02", "february": "02",
    "mar": "03", "march": "03",
    "apr": "04", "april": "04",
    "may": "05",
    "jun": "06", "june": "06",
    "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "sept": "09", "september": "09",
    "oct": "10", "october": "10",
    "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}

# "1st January 2024" / "December 2023" / "1 March 2026" -- day is optional.
_DATE_RE = re.compile(
    r"\b(?:(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+)?"
    r"(?P<month>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\s+"
    r"(?P<year>20\d{2})\b",
    re.IGNORECASE,
)
_YEAR_ONLY_RE = re.compile(r"\b(20\d{2})\b")

_WOMEN_RE = re.compile(r"\bwomen'?s?\b", re.IGNORECASE)
_MEN_RE = re.compile(r"\bmen'?s?\b", re.IGNORECASE)

# Order matters: more specific ("t20i"/"t20 international") must be checked
# before the generic "t20" (domestic, no "international") pattern.
_FORMAT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\btest\b", re.IGNORECASE), "test"),
    (re.compile(r"\bodi\b|one[- ]?day\b", re.IGNORECASE), "odi"),
    (re.compile(r"\bt20i\b|t20\s*international\b|twenty20\s*international\b", re.IGNORECASE), "t20i"),
    (re.compile(r"\bt20\b|twenty20\b", re.IGNORECASE), "t20"),
    (re.compile(r"\bmulti-?day\b", re.IGNORECASE), "multiday"),
]


def parse_gender(title: str) -> str:
    """Returns "women", "men" or "all" (no gender named -- e.g. a Code of Conduct)."""
    if _WOMEN_RE.search(title):
        return "women"
    if _MEN_RE.search(title):
        return "men"
    return "all"


def parse_format(title: str) -> str | None:
    """Returns "test" | "odi" | "t20i" | "t20" | "multiday", or None if the
    title doesn't name a format (e.g. a Code of Conduct)."""
    for pattern, fmt in _FORMAT_PATTERNS:
        if pattern.search(title):
            return fmt
    return None


def parse_effective_date(title: str) -> str | None:
    """Best-effort ISO-ish date from a title: "YYYY-MM-DD", "YYYY-MM" or "YYYY"."""
    m = _DATE_RE.search(title)
    if m:
        year = m.group("year")
        month = _MONTHS[m.group("month").lower()]
        day = m.group("day")
        if day:
            return f"{year}-{month}-{int(day):02d}"
        return f"{year}-{month}"
    m = _YEAR_ONLY_RE.search(title)
    return m.group(1) if m else None


# BCCI's domestic playing-conditions titles (e.g. "Playing Conditions for
# Men's Multi-day Matches") carry no date at all, but the PDF filename does,
# as a cricket season label: ".../1772193467492_2025-26_BCCI_Men_..._PC.pdf".
# IPL's titles are similarly bare (e.g. just "Match Playing Conditions"),
# but its filenames carry a plain year instead of a season range:
# ".../1775736835406_TATA_IPL_2026_Match_Playing_Conditions.pdf". Both
# filenames also start with an unrelated ~13-digit millisecond timestamp, so
# the year match requires a non-digit separator on both sides -- a plain
# `\b` word boundary doesn't help here since underscore counts as a "word"
# character in regex and so carries no boundary of its own.
# (BCCI's JSON also has a `publishDate` per document, which looked like a
# cleaner source -- but it's identical across every document on the page, a
# page-level timestamp rather than a per-document date, so it's not used.)
_SEASON_RE = re.compile(r"(?<!\d)(20\d{2})-(\d{2})(?!\d)")
_URL_YEAR_RE = re.compile(r"(?:^|[_/.-])(20\d{2})(?:[_/.-]|$)")


def parse_effective_date_from_url(url: str) -> str | None:
    """Best-effort date from a PDF's filename: a season's start year (e.g.
    "2025" from ".../2025-26_...pdf") or a plain year (e.g. "2026" from
    ".../IPL_2026_...pdf")."""
    m = _SEASON_RE.search(url)
    if m:
        return m.group(1)
    m = _URL_YEAR_RE.search(url)
    return m.group(1) if m else None


def make_doc_id(body: str, gender: str, competition: str, effective_from: str) -> str:
    """`competition` may be a bare suffix ("t20i") or the full, already
    body-prefixed value ("icc_t20i") -- either way the body isn't repeated
    in the result, e.g. body="icc" always yields "icc_mens_t20i_...", never
    "icc_mens_icc_t20i_...", whichever form the caller passes."""
    gender_slug = {"men": "mens", "women": "womens", "all": "all"}[gender]
    date_slug = effective_from.replace("-", "_")
    prefix = f"{body}_"
    suffix = competition[len(prefix) :] if competition.startswith(prefix) else competition
    return f"{body}_{gender_slug}_{suffix}_{date_slug}"
