"""Table of contents parsing."""

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from .models import TOCEntry
from .slicing import _fallback_title


def _make_toc_entry(title: str, href: str, children: list[TOCEntry] | None = None) -> TOCEntry:
    file_href, _, anchor = href.partition("#")
    return TOCEntry(title=title, href=href, file_href=file_href, anchor=anchor, children=children or [])


def parse_toc_recursive(toc_list) -> list[TOCEntry]:
    """
    Recursively parses the TOC structure from ebooklib.
    """
    result = []

    for item in toc_list:
        # ebooklib TOC items are either `Link`/`Section` objects or tuples (Section, [Children])
        if isinstance(item, tuple):
            section, children = item
            result.append(_make_toc_entry(section.title, section.href, parse_toc_recursive(children)))
        elif isinstance(item, (epub.Link, epub.Section)):
            result.append(_make_toc_entry(item.title, item.href))

    return result


def get_fallback_toc(book_obj) -> list[TOCEntry]:
    """
    If TOC is missing, build a flat one from the Spine, in reading order, titled by each file's first heading.
    """
    toc = []
    for spine_entry in book_obj.spine:
        idref = spine_entry[0] if isinstance(spine_entry, (tuple, list)) else spine_entry
        item = book_obj.get_item_with_id(idref)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        name = item.get_name()
        soup = BeautifulSoup(item.get_content().decode("utf-8", errors="ignore"), "html.parser")
        toc.append(_make_toc_entry(_fallback_title(soup, name), name))
    return toc


def _flatten_toc(entries: list[TOCEntry]) -> list[TOCEntry]:
    flat: list[TOCEntry] = []
    for e in entries:
        flat.append(e)
        if e.children:
            flat.extend(_flatten_toc(e.children))
    return flat
