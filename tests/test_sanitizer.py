import pytest
from bs4 import BeautifulSoup

from reader3 import _url_scheme, clean_html_content


def clean(html: str) -> str:
    return str(clean_html_content(BeautifulSoup(html, "html.parser")))


@pytest.mark.parametrize("value, expected", [
    ("javascript:alert(1)", "javascript"),
    ("JavaScript:alert(1)", "javascript"),
    ("  javascript:alert(1)", "javascript"),
    ("java\tscript:alert(1)", "javascript"),
    ("https://example.com", "https"),
    ("mailto:a@b.c", "mailto"),
    ("chapter.xhtml#x", None),
    ("#frag", None),
    ("../img/a.png", None),
])
def test_url_scheme(value, expected):
    assert _url_scheme(value) == expected


@pytest.mark.parametrize("html", [
    '<a href="javascript:alert(1)">x</a>',
    '<a href="JavaScript:alert(1)">x</a>',
    '<a href=" javascript:alert(1)">x</a>',
    '<a href="vbscript:msgbox(1)">x</a>',
    '<a href="data:text/html,<script>alert(1)</script>">x</a>',
    '<img src="data:text/html;base64,AAAA"/>',
])
def test_dangerous_urls_are_dropped(html):
    out = clean(html)
    assert "href=" not in out and "src=" not in out


@pytest.mark.parametrize("tag", ["script", "style", "iframe", "object", "embed", "form", "video", "base"])
def test_active_tags_are_removed(tag):
    assert f"<{tag}" not in clean(f"<p>keep</p><{tag}>x</{tag}>")


def test_event_handlers_and_comments_removed():
    out = clean('<p onclick="x()" ONMOUSEOVER="y()">hi</p><!-- c -->')
    assert "onclick" not in out.lower() and "onmouseover" not in out.lower()
    assert "<!--" not in out
    assert "hi" in out


def test_safe_content_is_preserved():
    html = ('<a href="https://example.com">a</a><a href="mailto:a@b.c">m</a>'
            '<a href="chapter.xhtml#x">r</a><img src="data:image/png;base64,AAAA"/>')
    out = clean(html)
    for kept in ("https://example.com", "mailto:a@b.c", "chapter.xhtml#x", "data:image/png"):
        assert kept in out
