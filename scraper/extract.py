"""Extract (title, pdf_url) candidates from a listing page's raw HTML.

Two shapes observed live on the actual sites (PROJECT_PLAN.md Phase 4 stop-
and-inspect): ICC's page has plain `<a href="...pdf">` anchors; BCCI's page
(a Next.js app) embeds its document list as escaped JSON inside a React
Server Components stream `<script>` payload, not as anchor tags -- both are
present in the static HTML with no JavaScript execution required, confirmed
by comparing a plain `requests.get()` against the rendered page.

(BCCI's JSON also carries a `publishDate` field per document, which looked
promising as a fallback date source for titles with no date of their own --
but live inspection showed it's identical across every document on the page
[a page-level "last touched in the CMS" timestamp, not a per-document
effective date], so it's not extracted here. scraper/metadata.py instead
falls back to the "YYYY-YY" season label BCCI embeds in the PDF filename
itself, which *is* genuinely per-document.)

IPL's own per-document pages (e.g. iplt20.com/playing-conditions, reached
from a hub page at iplt20.com/documents) turned out to use this exact same
embedded-JSON shape as BCCI -- same CMS platform, different tenant -- so
scraper/scrape.py's IPL crawl reuses `extract_bcci_links` directly rather
than needing a third parser or (as first assumed, before checking) a
headless browser.
"""
from __future__ import annotations

import json
import re

from bs4 import BeautifulSoup


def extract_icc_links(html: str) -> list[tuple[str, str]]:
    """Returns [(title, url), ...] for every <a href="...pdf"> on the page.

    The visible link text is the title; ICC occasionally also sets a `title`
    attribute with a shortened version, which we ignore in favour of the
    full link text for more reliable metadata parsing.
    """
    soup = BeautifulSoup(html, "html.parser")
    results: list[tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".pdf" not in href.lower():
            continue
        text = a.get_text(strip=True)
        if not text:
            continue
        results.append((text, href))
    return results


# BCCI's (and IPL's) Next.js RSC stream embeds each document as
# \"title\":\"...\",\"fileUrl\":\"https://...pdf\" -- the whole payload is a
# JSON-encoded string inside a <script>, so every quote is backslash-escaped
# at the HTML/JS source level. The two keys are directly adjacent (no
# intervening keys), so a tight non-wildcard pattern is safe from
# accidentally spanning two different documents. Each quote optionally
# allows a preceding backslash so this also matches a plain (unescaped)
# JSON encoding if the site's build ever changes.
#
# The captured content itself must tolerate JSON escapes too -- a title like
# "Players & Team Officials" (a plain "&") has a backslash that isn't a
# closing quote, so `[^"\\]*` alone would stop early and the whole pattern
# would fail to match. `(?:[^"\\]|\\[^"])*` instead treats backslash-plus-
# next-char as one atomic unit UNLESS that next char is itself a quote --
# because in this encoding every quote is backslash-escaped, including the
# ones that really do delimit the field, so `\"` must always be treated as
# the terminator (title values never contain a literal double-quote
# character) while `\u`, `\n` etc. are treated as escaped content.
# _unescape_json_fragment() then decodes the result properly (so "&"
# becomes "&", not the literal 6 characters).
_STRING_BODY = r'(?:[^"\\]|\\[^"])*'
_BCCI_DOC_RE = re.compile(
    rf'\\?"title\\?":\\?"(?P<title>{_STRING_BODY})\\?",\\?"fileUrl\\?":\\?"(?P<url>https:{_STRING_BODY}?\.pdf)\\?"'
)


def _unescape_json_fragment(raw: str) -> str:
    """Decodes standard JSON escapes (\\u0026, \\", \\\\, ...) in a fragment
    captured from inside a still-escaped JSON string."""
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        # A quote or backslash that survived unescaped (shouldn't happen
        # given the regex above, but fail soft rather than drop the match).
        return raw


def extract_bcci_links(html: str) -> list[tuple[str, str]]:
    """Returns [(title, url), ...] parsed out of BCCI's embedded document JSON."""
    return [
        (_unescape_json_fragment(m.group("title")), _unescape_json_fragment(m.group("url")))
        for m in _BCCI_DOC_RE.finditer(html)
    ]


_PARSERS = {
    "icc_html": extract_icc_links,
    "bcci_embedded_json": extract_bcci_links,
}


def extract_links(html: str, parser: str) -> list[tuple[str, str]]:
    try:
        fn = _PARSERS[parser]
    except KeyError:
        raise ValueError(f"unknown parser {parser!r}; expected one of {sorted(_PARSERS)}") from None
    return fn(html)
