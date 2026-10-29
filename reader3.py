"""
Parses an EPUB file into a structured object that can be used to serve the book via a web interface.
"""

import os
import pickle
import re
import shutil
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from urllib.parse import unquote

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, Comment
from bs4.element import Tag

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
    """The Master Object to be pickled."""
    metadata: BookMetadata
    spine: List[ChapterContent]  # The actual content (linear files)
    toc: List[TOCEntry]          # The navigation tree
    images: Dict[str, str]       # Map: original_path -> local_path

    # Meta info
    source_file: str
    processed_at: str
    version: str = "3.0"


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


def parse_toc_recursive(toc_list, depth=0) -> List[TOCEntry]:
    """
    Recursively parses the TOC structure from ebooklib.
    """
    result = []

    for item in toc_list:
        # ebooklib TOC items are either `Link` objects or tuples (Section, [Children])
        if isinstance(item, tuple):
            section, children = item
            entry = TOCEntry(
                title=section.title,
                href=section.href,
                file_href=section.href.split('#')[0],
                anchor=section.href.split('#')[1] if '#' in section.href else "",
                children=parse_toc_recursive(children, depth + 1)
            )
            result.append(entry)
        elif isinstance(item, epub.Link):
            entry = TOCEntry(
                title=item.title,
                href=item.href,
                file_href=item.href.split('#')[0],
                anchor=item.href.split('#')[1] if '#' in item.href else ""
            )
            result.append(entry)
        # Note: ebooklib sometimes returns direct Section objects without children
        elif isinstance(item, epub.Section):
             entry = TOCEntry(
                title=item.title,
                href=item.href,
                file_href=item.href.split('#')[0],
                anchor=item.href.split('#')[1] if '#' in item.href else ""
            )
             result.append(entry)

    return result


def get_fallback_toc(book_obj) -> List[TOCEntry]:
    """
    If TOC is missing, build a flat one from the Spine.
    """
    toc = []
    for item in book_obj.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            name = item.get_name()
            # Try to guess a title from the content or ID
            title = item.get_name().replace('.html', '').replace('.xhtml', '').replace('_', ' ').title()
            toc.append(TOCEntry(title=title, href=name, file_href=name, anchor=""))
    return toc


def extract_cover_image(book_obj, images_dir: str, image_map: Dict[str, str]) -> Optional[str]:
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
            original_fname = os.path.basename(cover_item.get_name())
            safe_fname = "cover_" + "".join([c for c in original_fname if c.isalpha() or c.isdigit() or c in '._-']).strip()
            local_path = os.path.join(images_dir, safe_fname)
            with open(local_path, 'wb') as f:
                f.write(cover_item.get_content())
            return f"images/{safe_fname}"
    except Exception as e:
        print(f"Method 1 (ITEM_COVER) failed: {e}")
    
    # Method 2: Check metadata for cover reference
    try:
        cover_meta = book_obj.get_metadata('OPF', 'cover')
        if cover_meta:
            cover_id = cover_meta[0][1].get('content') if cover_meta[0][1] else cover_meta[0][0]
            if cover_id:
                for item in book_obj.get_items():
                    if item.get_id() == cover_id or item.get_name().endswith(cover_id):
                        original_fname = os.path.basename(item.get_name())
                        safe_fname = "cover_" + "".join([c for c in original_fname if c.isalpha() or c.isdigit() or c in '._-']).strip()
                        local_path = os.path.join(images_dir, safe_fname)
                        with open(local_path, 'wb') as f:
                            f.write(item.get_content())
                        return f"images/{safe_fname}"
    except Exception as e:
        print(f"Method 2 (metadata cover) failed: {e}")
    
    # Method 3: Look for common cover image names
    cover_patterns = ['cover', 'Cover', 'COVER', 'front']
    for item in book_obj.get_items():
        if item.get_type() == ebooklib.ITEM_IMAGE:
            name = item.get_name().lower()
            if any(pattern.lower() in name for pattern in cover_patterns):
                original_fname = os.path.basename(item.get_name())
                safe_fname = "cover_" + "".join([c for c in original_fname if c.isalpha() or c.isdigit() or c in '._-']).strip()
                local_path = os.path.join(images_dir, safe_fname)
                with open(local_path, 'wb') as f:
                    f.write(item.get_content())
                return f"images/{safe_fname}"
    
    # Method 4: Get the first image as a fallback
    for item in book_obj.get_items():
        if item.get_type() == ebooklib.ITEM_IMAGE:
            original_fname = os.path.basename(item.get_name())
            safe_fname = "cover_" + "".join([c for c in original_fname if c.isalpha() or c.isdigit() or c in '._-']).strip()
            local_path = os.path.join(images_dir, safe_fname)
            with open(local_path, 'wb') as f:
                f.write(item.get_content())
            print("Warning: Using first image as cover (no explicit cover found)")
            return f"images/{safe_fname}"
    
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


def _find_anchor_tag(soup: BeautifulSoup, anchor: str, index: Optional[_AnchorIndex] = None):
    if not anchor:
        return None
    if index is not None:
        return index.find(anchor)
    tag = soup.find(id=anchor)
    if tag:
        return tag
    # Fallback for name="..."
    return soup.find(attrs={"name": anchor})


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
    print(f"Loading {epub_path}...")
    book = epub.read_epub(epub_path)

    # 2. Extract Metadata
    metadata = extract_metadata_robust(book)

    # 3. Prepare Output Directories (output_dir is expected to be a fresh staging dir)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # 4. Extract Cover Image First
    print("Extracting cover image...")
    image_map = {} # Key: internal_path, Value: local_relative_path
    cover_path = extract_cover_image(book, images_dir, image_map)
    if cover_path:
        metadata.cover_image = cover_path
        print(f"Cover extracted: {cover_path}")
    else:
        print("Warning: No cover image found")

    # 5. Extract Images & Build Map
    print("Extracting other images...")

    for item in book.get_items():
        if item.get_type() in (ebooklib.ITEM_IMAGE, ebooklib.ITEM_COVER):
            # Normalize filename
            original_fname = os.path.basename(item.get_name())
            # Sanitize filename for OS
            safe_fname = "".join([c for c in original_fname if c.isalpha() or c.isdigit() or c in '._-']).strip()

            # Save to disk
            local_path = os.path.join(images_dir, safe_fname)
            with open(local_path, 'wb') as f:
                f.write(item.get_content())

            # Map keys: We try both the full internal path and just the basename
            # to be robust against messy HTML src attributes
            rel_path = f"images/{safe_fname}"
            image_map[item.get_name()] = rel_path
            image_map[original_fname] = rel_path

    # 6. Process TOC
    print("Parsing Table of Contents...")
    toc_structure = parse_toc_recursive(book.toc)
    if not toc_structure:
        print("Warning: Empty TOC, building fallback from Spine...")
        toc_structure = get_fallback_toc(book)
    flat_toc = _flatten_toc(toc_structure)

    # 7. Process Content based on TOC (split by anchors for one-chapter-at-a-time)
    print("Processing chapters...")
    spine_chapters: List[ChapterContent] = []

    # Pre-parse all document files once
    doc_items: Dict[str, Any] = {}
    soups: Dict[str, BeautifulSoup] = {}
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            name = item.get_name()
            doc_items[name] = item
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
                    src_decoded = unquote(src)
                    filename = os.path.basename(src_decoded)
                    if src_decoded in image_map:
                        img[attr] = image_map[src_decoded]
                    elif filename in image_map:
                        img[attr] = image_map[filename]

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
    print("Rewriting internal links...")
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


def save_to_pickle(book: Book, output_dir: str):
    p_path = os.path.join(output_dir, 'book.pkl')
    with open(p_path, 'wb') as f:
        pickle.dump(book, f)
    print(f"Saved structured data to {p_path}")


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
        save_to_pickle(book, staging_dir)
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

if __name__ == "__main__":

    import sys
    if len(sys.argv) < 2:
        print("Usage: python reader3.py <file.epub>")
        sys.exit(1)

    epub_file = sys.argv[1]
    assert os.path.exists(epub_file), "File not found."
    book_name = os.path.splitext(os.path.basename(epub_file))[0]
    out_dir = os.path.join("library", f"{book_name}_data")

    book_obj = convert_epub(epub_file, out_dir)
    print("\n--- Summary ---")
    print(f"Title: {book_obj.metadata.title}")
    print(f"Authors: {', '.join(book_obj.metadata.authors)}")
    print(f"Physical Files (Spine): {len(book_obj.spine)}")
    print(f"TOC Root Items: {len(book_obj.toc)}")
    print(f"Cover Image Extracted: {'Yes' if book_obj.metadata.cover_image else 'No'}")
    print(f"Other Images extracted: {len(book_obj.images)}")
