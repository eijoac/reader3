"""HTML sanitising: drops active content and unsafe URLs from book markup."""

import re

from bs4 import BeautifulSoup, Comment

REMOVED_TAGS = [
    "script",
    "style",
    "iframe",
    "frame",
    "frameset",
    "object",
    "embed",
    "applet",
    "video",
    "audio",
    "source",
    "track",
    "canvas",
    "nav",
    "form",
    "button",
    "input",
    "select",
    "textarea",
    "link",
    "meta",
    "base",
    "foreignobject",
]

# Attributes whose value is a URL and must not carry an executable scheme.
URL_ATTRS = ("href", "src", "xlink:href", "action", "formaction", "poster", "data", "srcset")
SAFE_URL_SCHEMES = ("http", "https", "mailto")


def url_scheme(value: str) -> str | None:
    """Lowercased URL scheme as a browser would see it, or None for relative URLs and fragments."""
    # Browsers ignore whitespace/control characters inside the scheme (e.g. "java\tscript:")
    compact = "".join(ch for ch in value if ch.isprintable() and not ch.isspace()).lower()
    scheme_match = re.match(r"^([a-z][a-z0-9+.\-]*):", compact)
    return scheme_match.group(1) if scheme_match else None


def is_safe_url(value: str, attr: str, tag_name: str) -> bool:
    scheme = url_scheme(value)
    if scheme is None or scheme in SAFE_URL_SCHEMES:
        return True
    # Inline raster/vector images are inert when loaded through <img>
    compact = "".join(ch for ch in value if ch.isprintable() and not ch.isspace()).lower()
    return tag_name == "img" and attr == "src" and compact.startswith("data:image/")


def clean_html_content(soup: BeautifulSoup) -> BeautifulSoup:
    """
    Sanitizes untrusted EPUB markup before it is stored and rendered with `| safe`:
    drops active/embedding tags, comments, event-handler attributes and
    URLs with executable schemes (javascript:, vbscript:, data:text/html, ...).
    """

    for tag in soup(REMOVED_TAGS):
        tag.decompose()

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            attr_lower = attr.lower()
            if attr_lower.startswith("on"):
                del tag.attrs[attr]
            elif attr_lower in URL_ATTRS:
                value = tag.attrs[attr]
                if isinstance(value, list):
                    value = " ".join(value)
                if not is_safe_url(str(value), attr_lower, tag.name):
                    del tag.attrs[attr]

    return soup


def extract_plain_text(soup: BeautifulSoup) -> str:
    """Extract clean text for LLM/Search usage."""
    text = soup.get_text(separator=" ")
    # Collapse whitespace
    return " ".join(text.split())
