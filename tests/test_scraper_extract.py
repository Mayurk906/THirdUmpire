from scraper.extract import extract_bcci_links, extract_icc_links, extract_links

# Small, hand-trimmed but structurally faithful snippets of the real pages
# (PROJECT_PLAN.md Phase 4 stop-and-inspect), not full copies of the sites.

ICC_HTML = """
<html><body>
<div class="playing-conditions">
  <p><a target="_blank" href="https://images.icc-cricket.com/image/upload/prd/lm8owaz03i86m1eneb7m.pdf">ICC Men's Test Match Playing Conditions - June 2025</a></p>
  <p><a href="https://images.icc-cricket.com/image/upload/prd/d25dbgishkx0kijb4jeu.pdf">ICC Men's Standard ODI Playing Conditions Effective July 2025</a></p>
  <p><a href="/about/cricket/rules-and-regulations">Not a PDF link</a></p>
  <p><a href="https://images.icc-cricket.com/image/upload/prd/cectkrcjgimwsacl6bq6.pdf" title="short title">ICC Women's Test Match playing Conditions - July 2026</a></p>
</div>
</body></html>
"""

# A trimmed stand-in for BCCI's Next.js RSC stream payload: escaped JSON
# embedded inside a <script> tag, not real anchor tags.
BCCI_HTML = (
    r'<html><body><script>self.__next_f.push([1,"2:[\"$\",\"$L24\",null,{\"page\":{\"components\":['
    r'{\"items\":[{\"label\":\"x\",\"document\":{\"id\":\"3F3wflwnc6LyBrLVkWr25q\",\"type\":\"pdf\",'
    r'\"title\":\"Mens Standard Test Match Playing Conditions - Effective December 2023\",'
    r'\"fileUrl\":\"https://www.bcci.tv/cms/files/gtykgr9m2ngt/1akpCIV1QLrQgBxk57OdtI/1fc4674c7d6f6947710a3e9b3358f15a/Mens_Standard_Test_Match_Playing_Conditions-Effective_December_2023.pdf\",'
    r'\"category\":{\"name\":\"Playing Conditions\"}}},'
    r'{\"label\":\"y\",\"document\":{\"id\":\"abc\",\"type\":\"pdf\",'
    r'\"title\":\"Playing Conditions for Mens Multi-day Matches\",'
    r'\"fileUrl\":\"https://www.bcci.tv/cms/files/gtykgr9m2ngt/xxxx/1772193467492_2025-26_BCCI_Men_Multi_Day_PC.pdf\",'
    r'\"category\":{\"name\":\"Playing Conditions\"}}}'
    r']}]}}]\n"]);</script></body></html>'
)


def test_extract_icc_links_returns_title_and_url_pairs() -> None:
    links = extract_icc_links(ICC_HTML)
    assert (
        "ICC Men's Test Match Playing Conditions - June 2025",
        "https://images.icc-cricket.com/image/upload/prd/lm8owaz03i86m1eneb7m.pdf",
    ) in links
    assert (
        "ICC Men's Standard ODI Playing Conditions Effective July 2025",
        "https://images.icc-cricket.com/image/upload/prd/d25dbgishkx0kijb4jeu.pdf",
    ) in links


def test_extract_icc_links_ignores_non_pdf_anchors() -> None:
    links = extract_icc_links(ICC_HTML)
    assert not any("rules-and-regulations" in url for _, url in links)


def test_extract_icc_links_prefers_visible_text_over_title_attr() -> None:
    links = extract_icc_links(ICC_HTML)
    titles = [t for t, _ in links]
    assert "ICC Women's Test Match playing Conditions - July 2026" in titles
    assert "short title" not in titles


def test_extract_bcci_links_decodes_unicode_escape_in_title() -> None:
    # A real title seen on IPL's Code of Conduct sub-page (which reuses this
    # same parser): "Players & Team Officials" must decode to
    # "Players & Team Officials", not stop matching at the backslash.
    html = (
        r'<script>self.__next_f.push([1,"{\"items\":[{\"label\":\"x\",\"document\":{'
        r'\"title\":\"Code of Conduct for Players & Team Officials\",'
        r'\"fileUrl\":\"https://www.iplt20.com/cms/files/x/y/1774244043749_TATA_IPL_2026_Code_of_Conduct.pdf\",'
        r'\"category\":{\"name\":\"Regulations\"}}}]}"]);</script>'
    )
    links = extract_bcci_links(html)
    assert (
        "Code of Conduct for Players & Team Officials",
        "https://www.iplt20.com/cms/files/x/y/1774244043749_TATA_IPL_2026_Code_of_Conduct.pdf",
    ) in links


def test_extract_bcci_links_parses_embedded_json() -> None:
    links = extract_bcci_links(BCCI_HTML)
    assert (
        "Mens Standard Test Match Playing Conditions - Effective December 2023",
        "https://www.bcci.tv/cms/files/gtykgr9m2ngt/1akpCIV1QLrQgBxk57OdtI/1fc4674c7d6f6947710a3e9b3358f15a/Mens_Standard_Test_Match_Playing_Conditions-Effective_December_2023.pdf",
    ) in links
    assert (
        "Playing Conditions for Mens Multi-day Matches",
        "https://www.bcci.tv/cms/files/gtykgr9m2ngt/xxxx/1772193467492_2025-26_BCCI_Men_Multi_Day_PC.pdf",
    ) in links


def test_extract_links_dispatches_by_parser_name() -> None:
    assert extract_links(ICC_HTML, "icc_html") == extract_icc_links(ICC_HTML)
    assert extract_links(BCCI_HTML, "bcci_embedded_json") == extract_bcci_links(BCCI_HTML)


def test_extract_links_rejects_unknown_parser() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown parser"):
        extract_links(ICC_HTML, "nonexistent")
