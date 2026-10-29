"""
Parses an EPUB file into a structured object that can be used to serve the book via a web interface.
"""

import logging
import os
import json
import posixpath
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from urllib.parse import unquote

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, Comment
from bs4.element import Tag

log = logging.getLogger(__name__)

# --- Data structures ---

@dataclass
class ChapterContent:
    """
    Represents a physical file in the EPUB (Spine Item).
    A single file might contain multiple logical chapters (TOC entries).
    """
    id: str           # Internal ID (e.g., 'item_1')
    href: str         # Filename (e.g., 'part01.html')
    title: str        # Best guess title from file
    content: str      # Cleaned HTML with rewritten image paths
    text: str         # Plain text for search/LLM context
    order: int        # Linear reading order


@dataclass
class TOCEntry:
    """Represents a logical entry in the navigation sidebar."""
    title: str
    href: str         # original href (e.g., 'part01.html#chapter1')
    file_href: str    # just the filename (e.g., 'part01.html')
    anchor: str       # just the anchor (e.g., 'chapter1'), empty if none
    children: List['TOCEntry'] = field(default_factory=list)


@dataclass
class BookMetadata:
    """Metadata"""
    title: str
    language: str
    authors: List[str] = field(default_factory=list)
    description: Optional[str] = None
    publisher: Optional[str] = None
    date: Optional[str] = None
    identifiers: List[str] = field(default_factory=list)
    subjects: List[str] = field(default_factory=list)
    cover_image: Optional[str] = None  # Relative path to cover image


@dataclass
class Book:
    """The master object, stored as book.json."""
    metadata: BookMetadata
    spine: List[ChapterContent]  # The actual content (linear files)
    toc: List[TOCEntry]          # The navigation tree
    images: Dict[str, str]       # Map: original_path -> local_path

    # Meta info
    source_file: str
    processed_at: str


# --- Storage (book.json + meta.json) ---

# Bump when the stored layout changes; books with another version must be reprocessed.
FORMAT_VERSION = 4
BOOK_FILE = "book.json"
META_FILE = "meta.json"  # small summary, so the library page never has to load whole books
LEGACY_BOOK_FILE = "book.pkl"


def _toc_from_dict(d: dict) -> TOCEntry:
    return TOCEntry(**{**d, "children": [_toc_from_dict(c) for c in d.get("children", [])]})


def book_to_dict(book: Book) -> dict:
    return {"format_version": FORMAT_VERSION, **asdict(book)}


def book_from_dict(d: dict) -> Book:
    if d.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported book format version: {d.get('format_version')!r}")
    return Book(
        metadata=BookMetadata(**d["metadata"]),
        spine=[ChapterContent(**c) for c in d["spine"]],
        toc=[_toc_from_dict(t) for t in d["toc"]],
        images=dict(d["images"]),
        source_file=d["source_file"],
        processed_at=d["processed_at"],
    )


def book_summary(book: Book) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "title": book.metadata.title,
        "authors": list(book.metadata.authors),
        "chapters": len(book.spine),
        "cover_image": book.metadata.cover_image,
    }


def load_book_file(book_dir: str) -> Book:
    with open(os.path.join(book_dir, BOOK_FILE), "r", encoding="utf-8") as f:
        return book_from_dict(json.load(f))


def read_summary(book_dir: str) -> Optional[dict]:
    """The summary from meta.json, or None if it is missing, unreadable or from another format version."""
    try:
        with open(os.path.join(book_dir, META_FILE), "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format_version") == FORMAT_VERSION and isinstance(meta.get("title"), str):
            return meta
    except (OSError, ValueError):
        pass
    return None


def is_legacy_book(book_dir: str) -> bool:
    """True for folders written by older versions (book.pkl only), which need reprocessing."""
    return (os.path.isfile(os.path.join(book_dir, LEGACY_BOOK_FILE))
            and not os.path.isfile(os.path.join(book_dir, BOOK_FILE)))


# --- Utilities ---

REMOVED_TAGS = [
    'script', 'style', 'iframe', 'frame', 'frameset', 'object', 'embed', 'applet',
    'video', 'audio', 'source', 'track', 'canvas', 'nav', 'form', 'button',
    'input', 'select', 'textarea', 'link', 'meta', 'base', 'foreignobject',
]

# Attributes whose value is a URL and must not carry an executable scheme.
URL_ATTRS = ('href', 'src', 'xlink:href', 'action', 'formaction', 'poster', 'data', 'srcset')
SAFE_URL_SCHEMES = ('http', 'https', 'mailto')


def _url_scheme(value: str) -> Optional[str]:
    """Lowercased URL scheme as a browser would see it, or None for relative URLs and fragments."""
    # Browsers ignore whitespace/control characters inside the scheme (e.g. "java\tscript:")
    compact = ''.join(ch for ch in value if ch.isprintable() and not ch.isspace()).lower()
    scheme_match = re.match(r'^([a-z][a-z0-9+.\-]*):', compact)
    return scheme_match.group(1) if scheme_match else None


def _is_safe_url(value: str, attr: str, tag_name: str) -> bool:
    scheme = _url_scheme(value)
    if scheme is None or scheme in SAFE_URL_SCHEMES:
        return True
    # Inline raster/vector images are inert when loaded through <img>
    compact = ''.join(ch for ch in value if ch.isprintable() and not ch.isspace()).lower()
    return tag_name == 'img' and attr == 'src' and compact.startswith('data:image/')


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
            if attr_lower.startswith('on'):
                del tag.attrs[attr]
            elif attr_lower in URL_ATTRS:
                value = tag.attrs[attr]
                if isinstance(value, list):
                    value = ' '.join(value)
                if not _is_safe_url(str(value), attr_lower, tag.name):
                    del tag.attrs[attr]

    return soup


def extract_plain_text(soup: BeautifulSoup) -> str:
    """Extract clean text for LLM/Search usage."""
    text = soup.get_text(separator=' ')
    # Collapse whitespace
    return ' '.join(text.split())


def _make_toc_entry(title: str, href: str, children: Optional[List[TOCEntry]] = None) -> TOCEntry:
    file_href, _, anchor = href.partition('#')
    return TOCEntry(title=title, href=href, file_href=file_href, anchor=anchor, children=children or [])


def parse_toc_recursive(toc_list) -> List[TOCEntry]:
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


def get_fallback_toc(book_obj) -> List[TOCEntry]:
    """
    If TOC is missing, build a flat one from the Spine, in reading order.
    """
    toc = []
    for spine_entry in book_obj.spine:
        idref = spine_entry[0] if isinstance(spine_entry, (tuple, list)) else spine_entry
        item = book_obj.get_item_with_id(idref)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        name = item.get_name()
        title = name.replace('.html', '').replace('.xhtml', '').replace('_', ' ').title()
        toc.append(_make_toc_entry(title, name))
    return toc


def _unique_image_name(item_name: str, used_names: set, prefix: str = "") -> str:
    """
    Filesystem-safe name for an image, unique within the book.
    Images in different EPUB folders can share a basename (a/1.png, b/1.png), and
    different names can sanitize to the same string, so a numeric suffix is added on collision.
    """
    clean = "".join(c for c in os.path.basename(item_name) if c.isalnum() or c in '._-').strip() or "image"
    stem, ext = os.path.splitext(clean)
    name = prefix + clean
    counter = 2
    while name.lower() in used_names:  # case-insensitive: the output may live on a case-insensitive filesystem
        name = f"{prefix}{stem}_{counter}{ext}"
        counter += 1
    used_names.add(name.lower())
    return name


def _save_cover(item, images_dir: str, used_names: set) -> str:
    name = _unique_image_name(item.get_name(), used_names, prefix="cover_")
    with open(os.path.join(images_dir, name), 'wb') as f:
        f.write(item.get_content())
    return f"images/{name}"


def extract_cover_image(book_obj, images_dir: str, used_names: set) -> Optional[str]:
    """
    Extracts the cover image from the epub.
    Tries multiple methods to find the cover.
    Returns the relative path to the saved cover image.
    """
    # Method 1: Try to get cover from metadata
    try:
        cover_item = None
        for item in book_obj.get_items():
            if item.get_type() == ebooklib.ITEM_COVER:
                cover_item = item
                break
        
        if cover_item:
            return _save_cover(cover_item, images_dir, used_names)
    except Exception as e:
        log.warning("Cover lookup via ITEM_COVER failed: %s", e)
    
    # Method 2: Check metadata for cover reference
    try:
        cover_meta = book_obj.get_metadata('OPF', 'cover')
        if cover_meta:
            cover_id = cover_meta[0][1].get('content') if cover_meta[0][1] else cover_meta[0][0]
            if cover_id:
                for item in book_obj.get_items():
                    if item.get_id() == cover_id or item.get_name().endswith(cover_id):
                        return _save_cover(item, images_dir, used_names)
    except Exception as e:
        log.warning("Cover lookup via metadata failed: %s", e)
    
    # Method 3: Look for common cover image names
    cover_patterns = ['cover', 'Cover', 'COVER', 'front']
    for item in book_obj.get_items():
        if item.get_type() == ebooklib.ITEM_IMAGE:
            name = item.get_name().lower()
            if any(pattern.lower() in name for pattern in cover_patterns):
                return _save_cover(item, images_dir, used_names)
    
    # Method 4: Get the first image as a fallback
    for item in book_obj.get_items():
        if item.get_type() == ebooklib.ITEM_IMAGE:
            log.warning("Using first image as cover (no explicit cover found)")
            return _save_cover(item, images_dir, used_names)
    
    return None


def extract_metadata_robust(book_obj) -> BookMetadata:
    """
    Extracts metadata handling both single and list values.
    Note: cover_image is set separately in process_epub after image extraction.
    """
    def get_list(key):
        data = book_obj.get_metadata('DC', key)
        return [x[0] for x in data] if data else []

    def get_one(key):
        data = book_obj.get_metadata('DC', key)
        return data[0][0] if data else None

    return BookMetadata(
        title=get_one('title') or "Untitled",
        language=get_one('language') or "en",
        authors=get_list('creator'),
        description=get_one('description'),
        publisher=get_one('publisher'),
        date=get_one('date'),
        identifiers=get_list('identifier'),
        subjects=get_list('subject'),
        cover_image=None  # Set later
    )


# --- Main Conversion Logic ---

def _flatten_toc(entries: List[TOCEntry]) -> List[TOCEntry]:
    flat: List[TOCEntry] = []
    for e in entries:
        flat.append(e)
        if e.children:
            flat.extend(_flatten_toc(e.children))
    return flat


class _AnchorIndex:
    """
    One-pass lookup tables for a parsed document, so anchor and document-order
    queries don't rescan the whole DOM every time.
    """

    def __init__(self, soup: BeautifulSoup):
        self.soup = soup
        self._by_id: Dict[str, Tag] = {}
        self._all_by_id: Dict[str, List[Tag]] = {}
        self._by_name: Dict[str, Tag] = {}
        for tag in soup.find_all(True):
            tag_id = tag.get('id')
            if tag_id:
                self._all_by_id.setdefault(tag_id, []).append(tag)
                self._by_id.setdefault(tag_id, tag)
            tag_name = tag.get('name')
            if tag_name and tag_name not in self._by_name:
                self._by_name[tag_name] = tag
        self._positions: Optional[Dict[int, int]] = None
        self._body_positions: Optional[Dict[int, int]] = None

    def all_with_id(self, anchor: str) -> List[Tag]:
        return self._all_by_id.get(anchor, [])

    def find(self, anchor: Optional[str]) -> Optional[Tag]:
        """First element with id=anchor, else the first with name=anchor."""
        if not anchor:
            return None
        return self._by_id.get(anchor) or self._by_name.get(anchor)

    def position(self, tag: Optional[Tag]) -> int:
        """Document-order index of `tag`, or -1 if it is not in the document."""
        if self._positions is None:
            self._positions = {id(el): i for i, el in enumerate(self.soup.descendants)}
        return self._positions.get(id(tag), -1) if tag is not None else -1

    def body_position(self, tag: Optional[Tag]) -> int:
        """1-based position of `tag` among the body's descendants (past the end if absent)."""
        if tag is None:
            return 0
        if self._body_positions is None:
            body = self.soup.find('body') or self.soup
            self._body_positions = {id(el): i for i, el in enumerate(body.descendants, start=1)}
            self._body_positions[-1] = len(self._body_positions) + 1
        return self._body_positions.get(id(tag), self._body_positions[-1])


def _fallback_title(soup: BeautifulSoup, file_href: str) -> str:
    """Best-effort title for a document that has no TOC entry: first heading, else filename."""
    heading = soup.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    if heading:
        text = ' '.join(heading.get_text(separator=' ').split())
        if text:
            return text
    stem = os.path.splitext(os.path.basename(file_href))[0]
    return stem.replace('_', ' ').strip().title() or file_href


def _has_content(fragment_html: str) -> bool:
    """True if the fragment has visible text or an image (so it is worth a chapter of its own)."""
    frag = BeautifulSoup(fragment_html, 'html.parser')
    return bool(extract_plain_text(frag)) or frag.find(['img', 'svg', 'image']) is not None


def _slice_html_by_anchors(soup: BeautifulSoup, start_anchor: Optional[str], stop_anchors: List[str],
                           allow_empty: bool = False, index: Optional[_AnchorIndex] = None) -> str:
    """
    Returns an HTML fragment starting at `start_anchor` (or top of body when None)
    and ending right before the next occurrence of any of `stop_anchors`.
    Best-effort: works well when chapters start at headers or elements with id.
    With `allow_empty`, an empty slice is returned as "" instead of the whole body.
    """
    body = soup.find('body')
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
        # Prefer the header/section/div that contains the anchor as the visual start
        if start_tag.name in ['h1', 'h2', 'h3', 'h4', 'section', 'div', 'article']:
            start_node = start_tag
        else:
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
    stop_tag: Optional[Any] = None
    if start_node is None:
        # From beginning, choose the first anchor in doc order
        earliest = None
        for cand in stop_tag_candidates:
            if earliest is None or comes_before(cand, earliest):
                earliest = cand
        stop_tag = earliest
    else:
        for cand in stop_tag_candidates:
            if comes_before(start_node, cand):
                if stop_tag is None or comes_before(cand, stop_tag):
                    stop_tag = cand

    # Top-level body children that contain (as a descendant, not themselves) any of `tags`
    def containing_children(tags: List[Tag]) -> set:
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
        stop_id = stop_tag.get('id')
        stop_containers = containing_children([stop_tag] + (index.all_with_id(stop_id) if stop_id else []))

    # Collect HTML between start and stop
    html_parts: List[str] = []
    collecting = False if start_node is not None else True

    for child in body.children:
        # We operate at the top level of body to avoid deep nested slicing issues
        if not collecting and start_node is not None:
            # Start when we reach start_node or it is contained within this child
            if child == start_node or id(child) in start_containers:
                collecting = True

        if collecting:
            # Stop before stop_tag if it's this child or contained
            if stop_tag is not None:
                if child == stop_tag or id(child) in stop_containers:
                    break
            html_parts.append(str(child))

        # If starting at beginning, start collecting immediately
        if start_node is None:
            collecting = True

    # Fallback: if nothing collected, return full body contents
    if not html_parts:
        return "" if allow_empty else "".join(str(x) for x in body.contents)
    return "".join(html_parts)


def process_epub(epub_path: str, output_dir: str) -> Book:

    # 1. Load Book
    log.info("Loading %s...", epub_path)
    book = epub.read_epub(epub_path)

    # 2. Extract Metadata
    metadata = extract_metadata_robust(book)

    # 3. Prepare Output Directories (output_dir is expected to be a fresh staging dir)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # 4. Extract Cover Image First
    log.info("Extracting cover image...")
    image_map = {} # Key: internal_path, Value: local_relative_path
    used_image_names: set = set()
    basename_targets: Dict[str, List[str]] = {}
    cover_path = extract_cover_image(book, images_dir, used_image_names)
    if cover_path:
        metadata.cover_image = cover_path
        log.info("Cover extracted: %s", cover_path)
    else:
        log.warning("No cover image found")

    # 5. Extract Images & Build Map
    log.info("Extracting other images...")

    for item in book.get_items():
        if item.get_type() in (ebooklib.ITEM_IMAGE, ebooklib.ITEM_COVER):
            safe_fname = _unique_image_name(item.get_name(), used_image_names)

            local_path = os.path.join(images_dir, safe_fname)
            with open(local_path, 'wb') as f:
                f.write(item.get_content())

            rel_path = f"images/{safe_fname}"
            image_map[item.get_name()] = rel_path
            basename_targets.setdefault(os.path.basename(item.get_name()), []).append(rel_path)

    # Basename-only lookup is a fallback for sloppy src attributes; it is only safe when unambiguous
    for basename, targets in basename_targets.items():
        if len(targets) == 1:
            image_map.setdefault(basename, targets[0])

    # 6. Process TOC
    log.info("Parsing Table of Contents...")
    toc_structure = parse_toc_recursive(book.toc)
    if not toc_structure:
        log.warning("Empty TOC, building fallback from Spine...")
        toc_structure = get_fallback_toc(book)
    flat_toc = _flatten_toc(toc_structure)

    # 7. Process Content based on TOC (split by anchors for one-chapter-at-a-time)
    log.info("Processing chapters...")
    spine_chapters: List[ChapterContent] = []

    # Pre-parse all document files once
    soups: Dict[str, BeautifulSoup] = {}
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            name = item.get_name()
            raw_content = item.get_content().decode('utf-8', errors='ignore')
            soup = BeautifulSoup(raw_content, 'html.parser')

            # Sanitize first: the link rewrite below must only ever see already-vetted markup
            soup = clean_html_content(soup)

            # Fix Images
            # <img src> plus SVG <image xlink:href|href> (used by cover-page wrappers)
            for img in soup.find_all(['img', 'image']):
                attrs = ('src',) if img.name == 'img' else ('xlink:href', 'href')
                for attr in attrs:
                    src = img.get(attr, '')
                    if not src:
                        continue
                    src_decoded = unquote(src).split('#', 1)[0].split('?', 1)[0]
                    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), src_decoded))
                    for candidate in (resolved, src_decoded, posixpath.basename(src_decoded)):
                        if candidate in image_map:
                            img[attr] = image_map[candidate]
                            break

            # Fix Internal Links - rewrite to use a placeholder that will be replaced later
            # We'll use a special format: __INTERNAL_LINK__filename#anchor
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if not href or href.lstrip().startswith('#') or _url_scheme(href) is not None:
                    # External links (already vetted by the sanitizer), email, or same-page anchors
                    continue
                # This is likely an internal epub link
                # Store it in a special format for later processing
                href_decoded = unquote(href)
                # Percent-decoding can reveal a scheme the sanitizer couldn't see (e.g. "java%73cript:")
                if _url_scheme(href_decoded) is not None:
                    del link['href']
                    continue
                link['href'] = f"__INTERNAL_LINK__{href_decoded}"

            soups[name] = soup

    # Reading order comes from the spine; the TOC only supplies titles and split points.
    spine_files: List[str] = []
    for spine_entry in book.spine:
        idref = spine_entry[0] if isinstance(spine_entry, (tuple, list)) else spine_entry
        spine_item = book.get_item_with_id(idref)
        if spine_item is not None and spine_item.get_name() in soups and spine_item.get_name() not in spine_files:
            spine_files.append(spine_item.get_name())
    # Documents the spine doesn't list (but the TOC or manifest does) go last so nothing is dropped
    for name in [e.file_href for e in flat_toc] + list(soups):
        if name in soups and name not in spine_files:
            spine_files.append(name)

    toc_entries_by_file: Dict[str, List[TOCEntry]] = {}
    for entry in flat_toc:
        toc_entries_by_file.setdefault(entry.file_href, []).append(entry)

    # Build a mapping from href to spine index for link rewriting
    href_to_index: Dict[str, int] = {}
    
    # Also collect all anchors from all files for comprehensive mapping
    all_file_anchors: Dict[str, List[str]] = {}
    anchor_indexes: Dict[str, _AnchorIndex] = {}
    for file_name, soup in soups.items():
        anchors = []
        for tag in soup.find_all(id=True):
            anchors.append(tag['id'])
        for tag in soup.find_all(attrs={"name": True}):
            anchors.append(tag['name'])
        all_file_anchors[file_name] = anchors
        anchor_indexes[file_name] = _AnchorIndex(soup)
    
    # Generate chapters in spine order; within a file, split at TOC anchors in document order
    for file_href in spine_files:
        soup = soups[file_href]
        index = anchor_indexes[file_href]

        def anchor_position(anchor: Optional[str]) -> int:
            if anchor is None:
                return -1
            return index.position(index.find(anchor))

        # (anchor, title) pairs; anchors missing from the file are treated as the file start
        starts: List[Tuple[Optional[str], str]] = []
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
            fragment_html = _slice_html_by_anchors(soup, anchor, stop_anchors, allow_empty=anchor is None and bool(toc_anchors), index=index)
            # Only skip chapters we invented ourselves; TOC-listed ones must stay linkable
            if synthetic_start and anchor is None and not _has_content(fragment_html):
                continue

            # Extract plain text from the fragment
            frag_soup = BeautifulSoup(fragment_html, 'html.parser')
            text = extract_plain_text(frag_soup)

            ch_href = f"{file_href}#{anchor}" if anchor else file_href
            chapter = ChapterContent(
                id=ch_href,
                href=ch_href,
                title=title or _fallback_title(frag_soup, file_href),
                content=fragment_html,
                text=text,
                order=len(spine_chapters)
            )
            spine_chapters.append(chapter)

            current_index = len(spine_chapters) - 1

            href_to_index[ch_href] = current_index
            # Links to a bare file go to its first chapter
            href_to_index.setdefault(file_href, current_index)
            href_to_index.setdefault(os.path.basename(file_href), current_index)
            if anchor:
                href_to_index[os.path.basename(file_href) + '#' + anchor] = current_index

    # Assign non-TOC anchors to nearest preceding chapter in the same file
    chapters_by_file: Dict[str, List[Tuple[Optional[str], int]]] = {}
    for i, ch in enumerate(spine_chapters):
        ch_file, _, ch_anchor = ch.href.partition('#')
        chapters_by_file.setdefault(ch_file, []).append((ch_anchor or None, i))

    for file_name, soup in soups.items():
        index = anchor_indexes[file_name]
        # Collect chapter starts (by anchor) for this file and resolve their DOM positions
        chapter_starts = chapters_by_file.get(file_name, [])  # (anchor, index)

        # Resolve tags and sort starts by DOM order
        start_tags: List[Tuple[Optional[str], int, Optional[Tag], int]] = []  # (anchor, idx, tag, pos)
        for anchor_id, idx in chapter_starts:
            if anchor_id is None:
                start_tags.append((None, idx, None, 0))
            else:
                tag = index.find(anchor_id)
                start_tags.append((anchor_id, idx, tag, index.body_position(tag)))
        start_tags.sort(key=lambda t: t[3])

        # For each discovered anchor in file, assign chapter index by containment:
        # 1) If exact match to a start anchor, choose that chapter.
        # 2) Else choose the nearest PREVIOUS start (largest s_pos <= a_pos),
        #    which means the anchor lies within that chapter's slice.
        for anchor_id in all_file_anchors.get(file_name, []):
            a_tag = index.find(anchor_id)
            if not a_tag:
                continue
            chosen_idx = None
            a_pos = index.body_position(a_tag)
            # Exact match first
            for s_anchor, idx, s_tag, s_pos in start_tags:
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
                href_to_index[os.path.basename(file_name) + '#' + anchor_id] = chosen_idx

    # Now rewrite all internal links in the chapters to use indices
    log.info("Rewriting internal links...")
    link_pattern = re.compile(r'href=(["\'])__INTERNAL_LINK__(.*?)\1', re.DOTALL)

    def rewrite_link(m: re.Match) -> str:
        quote, old_href = m.group(1), m.group(2)
        chapter_idx = href_to_index.get(old_href)
        if chapter_idx is None:
            return m.group(0)
        # Preserve the fragment if present
        frag = '#' + old_href.split('#', 1)[1] if '#' in old_href else ''
        return f'href={quote}__CHAPTER__{chapter_idx}{frag}{quote}'

    for chapter in spine_chapters:
        # Replace __INTERNAL_LINK__ markers with actual indices (single pass over the content)
        content = link_pattern.sub(rewrite_link, chapter.content)

        # Also handle any remaining __INTERNAL_LINK__ that didn't match
        # Just strip the marker and leave as relative (better than breaking)
        content = content.replace('__INTERNAL_LINK__', '')
        chapter.content = content

    # 8. Final Assembly
    final_book = Book(
        metadata=metadata,
        spine=spine_chapters,
        toc=toc_structure,
        images=image_map,
        source_file=os.path.basename(epub_path),
        processed_at=datetime.now().isoformat()
    )

    return final_book


def save_book(book: Book, output_dir: str):
    for name, data in ((BOOK_FILE, book_to_dict(book)), (META_FILE, book_summary(book))):
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    log.info("Saved structured data to %s", os.path.join(output_dir, BOOK_FILE))


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
    return book


# --- CLI ---

LIBRARY_DIR = "library"
USAGE = """Usage:
  reader3.py <file.epub>          convert a book into the library
  reader3.py list                 list converted books
  reader3.py reprocess <name>|--all   re-convert from library/<name>.epub
  reader3.py remove <name>        delete a converted book (the .epub is kept)"""


def _book_names(library_dir: str) -> List[str]:
    if not os.path.isdir(library_dir):
        return []
    return sorted(
        d[:-len("_data")] for d in os.listdir(library_dir)
        if d.endswith("_data") and (
            os.path.isfile(os.path.join(library_dir, d, BOOK_FILE)) or is_legacy_book(os.path.join(library_dir, d)))
    )


def _print_summary(book_obj: Book):
    print("\n--- Summary ---")
    print(f"Title: {book_obj.metadata.title}")
    print(f"Authors: {', '.join(book_obj.metadata.authors)}")
    print(f"Physical Files (Spine): {len(book_obj.spine)}")
    print(f"TOC Root Items: {len(book_obj.toc)}")
    print(f"Cover Image Extracted: {'Yes' if book_obj.metadata.cover_image else 'No'}")
    print(f"Other Images extracted: {len(book_obj.images)}")


def _convert_cli(epub_file: str, library_dir: str, summary: bool = True) -> bool:
    book_name = os.path.splitext(os.path.basename(epub_file))[0]
    out_dir = os.path.join(library_dir, f"{book_name}_data")
    try:
        book_obj = convert_epub(epub_file, out_dir)
    except Exception as e:
        log.debug("Conversion failed", exc_info=True)
        print(f"Error: could not convert {epub_file}: {e}", file=sys.stderr)
        print("Any previously converted copy of this book was left untouched.", file=sys.stderr)
        return False
    if summary:
        _print_summary(book_obj)
    return True


def _cmd_list(library_dir: str) -> int:
    names = _book_names(library_dir)
    if not names:
        print(f"No books in {library_dir}/")
        return 0
    for name in names:
        source = "epub" if os.path.isfile(os.path.join(library_dir, f"{name}.epub")) else "no epub"
        data_dir = os.path.join(library_dir, f"{name}_data")
        meta = read_summary(data_dir)
        if meta:
            title, authors = meta["title"], ", ".join(meta["authors"])
        else:
            title, authors = "(old or unreadable format - run reprocess)", ""
        print(f"{name}  [{source}]  {title}" + (f" - {authors}" if authors else ""))
    return 0


def _cmd_reprocess(args: List[str], library_dir: str) -> int:
    if len(args) != 1:
        print(USAGE, file=sys.stderr)
        return 1
    names = _book_names(library_dir) if args[0] == "--all" else [args[0]]
    if not names:
        print(f"No books in {library_dir}/")
        return 0
    failed = 0
    for name in names:
        epub = os.path.join(library_dir, f"{name}.epub")
        if not os.path.isfile(epub):
            print(f"Error: {epub} not found; cannot reprocess {name}", file=sys.stderr)
            failed += 1
        elif not _convert_cli(epub, library_dir, summary=False):
            failed += 1
    if len(names) > 1:
        print(f"Reprocessed {len(names) - failed} of {len(names)} books")
    return 1 if failed else 0


def _cmd_remove(args: List[str], library_dir: str) -> int:
    if len(args) != 1:
        print(USAGE, file=sys.stderr)
        return 1
    name = args[0]
    # Only plain names: no separators or traversal
    if name in ("", ".", "..") or os.sep in name or (os.altsep and os.altsep in name):
        print(f"Error: invalid book name: {name!r}", file=sys.stderr)
        return 1
    data_dir = os.path.join(library_dir, f"{name}_data")
    if not (os.path.isfile(os.path.join(data_dir, BOOK_FILE)) or is_legacy_book(data_dir)):
        print(f"Error: no converted book named {name!r}", file=sys.stderr)
        return 1
    shutil.rmtree(data_dir)
    print(f"Removed {data_dir} (kept {name}.epub if present)")
    return 0


def main(argv: List[str], library_dir: str = LIBRARY_DIR) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = argv[1:]
    if not args:
        print(USAGE, file=sys.stderr)
        return 1
    command, rest = args[0], args[1:]
    if command == "list":
        return _cmd_list(library_dir)
    if command == "reprocess":
        return _cmd_reprocess(rest, library_dir)
    if command == "remove":
        return _cmd_remove(rest, library_dir)

    if not os.path.isfile(command):
        print(f"Error: file not found: {command}", file=sys.stderr)
        return 1
    return 0 if _convert_cli(command, library_dir) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
