"""EPUB -> Book conversion and atomic writing of the result."""

import json
import logging
import os
import re
import shutil
from datetime import datetime
from urllib.parse import unquote

import ebooklib
from bs4 import BeautifulSoup
from bs4.element import Tag
from ebooklib import epub

from .images import extract_cover_image, extract_images, rewrite_image_sources
from .metadata import extract_metadata_robust
from .models import Book, ChapterContent, TOCEntry
from .sanitize import clean_html_content, extract_plain_text, url_scheme
from .slicing import _AnchorIndex, _fallback_title, _has_content, _slice_html_by_anchors
from .storage import BOOK_FILE, META_FILE, book_summary, book_to_dict
from .toc import _flatten_toc, get_fallback_toc, parse_toc_recursive

log = logging.getLogger(__name__)

INTERNAL_LINK_MARK = "__INTERNAL_LINK__"
INTERNAL_LINK_RE = re.compile(r'href=(["\'])' + INTERNAL_LINK_MARK + r"(.*?)\1", re.DOTALL)


def _load_documents(book: epub.EpubBook, image_map: dict[str, str]) -> dict[str, BeautifulSoup]:
    """Parses every document once: sanitised, with image sources and internal links rewritten."""
    soups: dict[str, BeautifulSoup] = {}
    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        name = item.get_name()
        raw_content = item.get_content().decode("utf-8", errors="ignore")
        # Sanitize first: the rewrites below must only ever see already-vetted markup
        soup = clean_html_content(BeautifulSoup(raw_content, "html.parser"))
        rewrite_image_sources(soup, name, image_map)
        _mark_internal_links(soup)
        soups[name] = soup
    return soups


def _mark_internal_links(soup: BeautifulSoup) -> None:
    """Prefixes epub-internal hrefs with a marker; they become chapter links once indices are known."""
    for link in soup.find_all("a"):
        href = link.get("href", "")
        if not href or href.lstrip().startswith("#") or url_scheme(href) is not None:
            # External links (already vetted by the sanitizer), email, or same-page anchors
            continue
        href_decoded = unquote(href)
        # Percent-decoding can reveal a scheme the sanitizer couldn't see (e.g. "java%73cript:")
        if url_scheme(href_decoded) is not None:
            del link["href"]
            continue
        link["href"] = f"{INTERNAL_LINK_MARK}{href_decoded}"


def _reading_order(book: epub.EpubBook, flat_toc: list[TOCEntry], soups: dict[str, BeautifulSoup]) -> list[str]:
    """Document names in reading order. The spine decides; the TOC only supplies titles and split points."""
    files: list[str] = []
    for spine_entry in book.spine:
        idref = spine_entry[0] if isinstance(spine_entry, (tuple, list)) else spine_entry
        spine_item = book.get_item_with_id(idref)
        if spine_item is not None and spine_item.get_name() in soups and spine_item.get_name() not in files:
            files.append(spine_item.get_name())
    # Documents the spine doesn't list (but the TOC or manifest does) go last so nothing is dropped
    for name in [e.file_href for e in flat_toc] + list(soups):
        if name in soups and name not in files:
            files.append(name)
    return files


def _document_anchors(soup: BeautifulSoup) -> list[str]:
    anchors = [tag["id"] for tag in soup.find_all(id=True)]
    anchors += [tag["name"] for tag in soup.find_all(attrs={"name": True})]
    return anchors


def _build_chapters(
    files: list[str],
    soups: dict[str, BeautifulSoup],
    anchor_indexes: dict[str, _AnchorIndex],
    flat_toc: list[TOCEntry],
) -> tuple[list[ChapterContent], dict[str, int]]:
    """
    Generates chapters in reading order; within a file, splits at TOC anchors in document order.
    Also returns the link map {href -> chapter index} for everything that opens a chapter directly.
    """
    toc_entries_by_file: dict[str, list[TOCEntry]] = {}
    for entry in flat_toc:
        toc_entries_by_file.setdefault(entry.file_href, []).append(entry)

    chapters: list[ChapterContent] = []
    href_to_index: dict[str, int] = {}

    for file_href in files:
        soup = soups[file_href]
        index = anchor_indexes[file_href]

        def anchor_position(anchor: str | None, index: _AnchorIndex = index) -> int:
            if anchor is None:
                return -1
            return index.position(index.find(anchor))

        # (anchor, title) pairs; anchors missing from the file are treated as the file start
        starts: list[tuple[str | None, str]] = []
        seen_anchors = set()
        for entry in toc_entries_by_file.get(file_href, []):
            anchor = entry.anchor if entry.anchor and index.find(entry.anchor) else None
            if anchor in seen_anchors:
                continue
            seen_anchors.add(anchor)
            starts.append((anchor, entry.title))
        starts.sort(key=lambda s: anchor_position(s[0]))

        toc_anchors = [a for a, _ in starts if a]
        synthetic_start = False
        if not starts:
            starts = [(None, "")]
            synthetic_start = True
        elif None not in seen_anchors:
            # Content before the first TOC anchor still belongs to the book
            leading_html = _slice_html_by_anchors(soup, None, toc_anchors, allow_empty=True, index=index)
            if _has_content(leading_html):
                starts.insert(0, (None, ""))
                synthetic_start = True

        for anchor, title in starts:
            stop_anchors = [a for a in toc_anchors if a != anchor]
            fragment_html = _slice_html_by_anchors(
                soup, anchor, stop_anchors, allow_empty=anchor is None and bool(toc_anchors), index=index
            )
            # Only skip chapters we invented ourselves; TOC-listed ones must stay linkable
            if synthetic_start and anchor is None and not _has_content(fragment_html):
                continue

            frag_soup = BeautifulSoup(fragment_html, "html.parser")
            ch_href = f"{file_href}#{anchor}" if anchor else file_href
            chapters.append(
                ChapterContent(
                    id=ch_href,
                    href=ch_href,
                    title=title or _fallback_title(frag_soup, file_href),
                    content=fragment_html,
                    text=extract_plain_text(frag_soup),
                    order=len(chapters),
                )
            )

            current_index = len(chapters) - 1
            href_to_index[ch_href] = current_index
            # Links to a bare file go to its first chapter
            href_to_index.setdefault(file_href, current_index)
            href_to_index.setdefault(os.path.basename(file_href), current_index)
            if anchor:
                href_to_index[os.path.basename(file_href) + "#" + anchor] = current_index

    return chapters, href_to_index


def _map_other_anchors(
    chapters: list[ChapterContent],
    soups: dict[str, BeautifulSoup],
    anchor_indexes: dict[str, _AnchorIndex],
    href_to_index: dict[str, int],
) -> None:
    """Links to anchors that don't start a chapter resolve to the chapter whose slice contains them."""
    chapters_by_file: dict[str, list[tuple[str | None, int]]] = {}
    for i, ch in enumerate(chapters):
        ch_file, _, ch_anchor = ch.href.partition("#")
        chapters_by_file.setdefault(ch_file, []).append((ch_anchor or None, i))

    for file_name, soup in soups.items():
        index = anchor_indexes[file_name]

        # Chapter starts of this file, with their DOM positions, in document order
        start_tags: list[tuple[str | None, int, Tag | None, int]] = []  # (anchor, idx, tag, pos)
        for anchor_id, idx in chapters_by_file.get(file_name, []):
            if anchor_id is None:
                start_tags.append((None, idx, None, 0))
            else:
                tag = index.find(anchor_id)
                start_tags.append((anchor_id, idx, tag, index.body_position(tag)))
        start_tags.sort(key=lambda t: t[3])

        for anchor_id in _document_anchors(soup):
            a_tag = index.find(anchor_id)
            if not a_tag:
                continue
            chosen_idx = None
            a_pos = index.body_position(a_tag)
            # Exact match to a chapter start first
            for s_anchor, idx, s_tag, _ in start_tags:
                if s_tag is not None and s_anchor == anchor_id:
                    chosen_idx = idx
                    break
            if chosen_idx is None:
                # Consider nearest next and previous; prefer next if very close (illustrations at chapter start)
                next_candidates = [(s_pos, idx) for _, idx, _, s_pos in start_tags if s_pos >= a_pos]
                prev_candidates = [(s_pos, idx) for _, idx, _, s_pos in start_tags if s_pos <= a_pos]
                threshold_forward = 20  # DOM steps considered "at start"
                if next_candidates:
                    next_pos, next_idx = min(next_candidates, key=lambda t: t[0])
                else:
                    next_pos, next_idx = (None, None)
                if prev_candidates:
                    prev_pos, prev_idx = max(prev_candidates, key=lambda t: t[0])
                else:
                    prev_pos, prev_idx = (None, None)
                if next_pos is not None and (prev_pos is None or (next_pos - a_pos) <= threshold_forward):
                    chosen_idx = next_idx
                elif prev_idx is not None:
                    chosen_idx = prev_idx
            if chosen_idx is not None:
                href_to_index[f"{file_name}#{anchor_id}"] = chosen_idx
                href_to_index[os.path.basename(file_name) + "#" + anchor_id] = chosen_idx


def _rewrite_internal_links(chapters: list[ChapterContent], href_to_index: dict[str, int]) -> None:
    """Replaces the internal-link markers with __CHAPTER__<index>[#fragment] placeholders (one pass per chapter)."""

    def rewrite_link(m: re.Match) -> str:
        quote, old_href = m.group(1), m.group(2)
        chapter_idx = href_to_index.get(old_href)
        if chapter_idx is None:
            return m.group(0)
        frag = "#" + old_href.split("#", 1)[1] if "#" in old_href else ""
        return f"href={quote}__CHAPTER__{chapter_idx}{frag}{quote}"

    for chapter in chapters:
        content = INTERNAL_LINK_RE.sub(rewrite_link, chapter.content)
        # Unresolvable links: just strip the marker and leave them relative (better than breaking)
        chapter.content = content.replace(INTERNAL_LINK_MARK, "")


def process_epub(epub_path: str, output_dir: str) -> Book:
    """Builds the Book from an EPUB; images go to `output_dir/images` (expected to be a fresh directory)."""
    log.info("Loading %s...", epub_path)
    book = epub.read_epub(epub_path)
    metadata = extract_metadata_robust(book)

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    log.info("Extracting cover image...")
    used_image_names: set = set()
    cover_path = extract_cover_image(book, images_dir, used_image_names)
    if cover_path:
        metadata.cover_image = cover_path
        log.info("Cover extracted: %s", cover_path)
    else:
        log.warning("No cover image found")

    log.info("Extracting other images...")
    image_map = extract_images(book, images_dir, used_image_names)

    log.info("Parsing Table of Contents...")
    toc_structure = parse_toc_recursive(book.toc)
    if not toc_structure:
        log.warning("Empty TOC, building fallback from Spine...")
        toc_structure = get_fallback_toc(book)
    flat_toc = _flatten_toc(toc_structure)

    log.info("Processing chapters...")
    soups = _load_documents(book, image_map)
    anchor_indexes = {name: _AnchorIndex(soup) for name, soup in soups.items()}
    files = _reading_order(book, flat_toc, soups)
    chapters, href_to_index = _build_chapters(files, soups, anchor_indexes, flat_toc)
    _map_other_anchors(chapters, soups, anchor_indexes, href_to_index)

    log.info("Rewriting internal links...")
    _rewrite_internal_links(chapters, href_to_index)

    return Book(
        metadata=metadata,
        spine=chapters,
        toc=toc_structure,
        images=image_map,
        source_file=os.path.basename(epub_path),
        processed_at=datetime.now().isoformat(),
    )


def save_book(book: Book, output_dir: str):
    for name, data in ((BOOK_FILE, book_to_dict(book)), (META_FILE, book_summary(book))):
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)


def convert_epub(epub_path: str, output_dir: str) -> Book:
    """
    Converts an EPUB into `output_dir`, replacing any previous conversion only
    after the new one has been fully built and saved. On failure, the existing
    output (if any) is left untouched.
    """
    parent = os.path.dirname(os.path.abspath(output_dir))
    os.makedirs(parent, exist_ok=True)

    # Hidden names that don't end in "_data", so the server never lists them as books
    base = os.path.basename(os.path.normpath(output_dir))
    staging_dir = os.path.join(parent, f".{base}.staging")
    backup_dir = os.path.join(parent, f".{base}.backup")
    for stale in (staging_dir, backup_dir):
        shutil.rmtree(stale, ignore_errors=True)

    try:
        book = process_epub(epub_path, staging_dir)
        save_book(book, staging_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    had_previous = os.path.exists(output_dir)
    try:
        if had_previous:
            os.rename(output_dir, backup_dir)
        os.rename(staging_dir, output_dir)
    except BaseException:
        if had_previous and os.path.exists(backup_dir) and not os.path.exists(output_dir):
            os.rename(backup_dir, output_dir)
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    shutil.rmtree(backup_dir, ignore_errors=True)
    log.info("Saved structured data to %s", os.path.join(output_dir, BOOK_FILE))
    return book
