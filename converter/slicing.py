"""Splitting a document into chapters at TOC anchors."""

import os
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import Tag

from .sanitize import extract_plain_text


class _AnchorIndex:
    """
    One-pass lookup tables for a parsed document, so anchor and document-order
    queries don't rescan the whole DOM every time.
    """

    def __init__(self, soup: BeautifulSoup):
        self.soup = soup
        self._by_id: dict[str, Tag] = {}
        self._all_by_id: dict[str, list[Tag]] = {}
        self._by_name: dict[str, Tag] = {}
        for tag in soup.find_all(True):
            tag_id = tag.get("id")
            if tag_id:
                self._all_by_id.setdefault(tag_id, []).append(tag)
                self._by_id.setdefault(tag_id, tag)
            tag_name = tag.get("name")
            if tag_name and tag_name not in self._by_name:
                self._by_name[tag_name] = tag
        self._positions: dict[int, int] | None = None
        self._body_positions: dict[int, int] | None = None

    def all_with_id(self, anchor: str) -> list[Tag]:
        return self._all_by_id.get(anchor, [])

    def find(self, anchor: str | None) -> Tag | None:
        """First element with id=anchor, else the first with name=anchor."""
        if not anchor:
            return None
        return self._by_id.get(anchor) or self._by_name.get(anchor)

    def position(self, tag: Tag | None) -> int:
        """Document-order index of `tag`, or -1 if it is not in the document."""
        if self._positions is None:
            self._positions = {id(el): i for i, el in enumerate(self.soup.descendants)}
        return self._positions.get(id(tag), -1) if tag is not None else -1

    def body_position(self, tag: Tag | None) -> int:
        """1-based position of `tag` among the body's descendants (past the end if absent)."""
        if tag is None:
            return 0
        if self._body_positions is None:
            body = self.soup.find("body") or self.soup
            self._body_positions = {id(el): i for i, el in enumerate(body.descendants, start=1)}
            self._body_positions[-1] = len(self._body_positions) + 1
        return self._body_positions.get(id(tag), self._body_positions[-1])


def _fallback_title(soup: BeautifulSoup, file_href: str) -> str:
    """Best-effort title for a document that has no TOC entry: first heading, else filename."""
    heading = soup.find(["h1", "h2", "h3", "h4", "h5", "h6"])
    if heading:
        text = " ".join(heading.get_text(separator=" ").split())
        if text:
            return text
    stem = os.path.splitext(os.path.basename(file_href))[0]
    return stem.replace("_", " ").strip().title() or file_href


def _has_content(fragment_html: str) -> bool:
    """True if the fragment has visible text or an image (so it is worth a chapter of its own)."""
    frag = BeautifulSoup(fragment_html, "html.parser")
    return bool(extract_plain_text(frag)) or frag.find(["img", "svg", "image"]) is not None


def _slice_html_by_anchors(
    soup: BeautifulSoup,
    start_anchor: str | None,
    stop_anchors: list[str],
    allow_empty: bool = False,
    index: _AnchorIndex | None = None,
) -> str:
    """
    Returns an HTML fragment starting at `start_anchor` (or top of body when None)
    and ending right before the next occurrence of any of `stop_anchors`.
    Best-effort: works well when chapters start at headers or elements with id.
    With `allow_empty`, an empty slice is returned as "" instead of the whole body.
    """
    body = soup.find("body")
    if not body:
        return str(soup)
    if index is None:
        index = _AnchorIndex(soup)

    # If no start anchor: start at beginning of body
    if not start_anchor:
        start_node = None
    else:
        start_tag = index.find(start_anchor)
        if not start_tag:
            # Anchor not found; return entire body as fallback
            return "".join(str(x) for x in body.contents)
        start_node = start_tag

    # Determine the next stop tag (first that appears after start)
    stop_tag_candidates = []
    for a in stop_anchors:
        t = index.find(a)
        if t:
            stop_tag_candidates.append(t)

    # True when b_tag appears strictly after a_tag in document order
    def comes_before(a_tag, b_tag) -> bool:
        if a_tag is None or b_tag is None:
            return False
        pos_a, pos_b = index.position(a_tag), index.position(b_tag)
        return pos_a >= 0 and pos_b > pos_a

    # Pick the earliest stop tag that comes after start
    stop_tag: Any | None = None
    if start_node is None:
        # From beginning, choose the first anchor in doc order
        earliest = None
        for cand in stop_tag_candidates:
            if earliest is None or comes_before(cand, earliest):
                earliest = cand
        stop_tag = earliest
    else:
        for cand in stop_tag_candidates:
            if comes_before(start_node, cand) and (stop_tag is None or comes_before(cand, stop_tag)):
                stop_tag = cand

    # Top-level body children that contain (as a descendant, not themselves) any of `tags`
    def containing_children(tags: list[Tag]) -> set:
        result = set()
        for t in tags:
            node = t.parent
            while node is not None and node is not body:
                if node.parent is body:
                    result.add(id(node))
                    break
                node = node.parent
        return result

    start_containers = containing_children(index.all_with_id(start_anchor)) if start_node is not None else set()
    stop_containers: set = set()
    if stop_tag is not None:
        stop_id = stop_tag.get("id")
        stop_containers = containing_children([stop_tag] + (index.all_with_id(stop_id) if stop_id else []))

    # Collect HTML between start and stop
    html_parts: list[str] = []
    collecting = start_node is None

    for child in body.children:
        # We operate at the top level of body to avoid deep nested slicing issues
        # Start when we reach start_node or it is contained within this child
        if not collecting and (child == start_node or id(child) in start_containers):
            collecting = True

        if collecting:
            # Stop before stop_tag if it's this child or contained
            if stop_tag is not None and (child == stop_tag or id(child) in stop_containers):
                break
            html_parts.append(str(child))

        # If starting at beginning, start collecting immediately
        if start_node is None:
            collecting = True

    # Fallback: if nothing collected, return full body contents
    if not html_parts:
        return "" if allow_empty else "".join(str(x) for x in body.contents)
    return "".join(html_parts)
