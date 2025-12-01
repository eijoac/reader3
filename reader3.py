"""
Parses an EPUB file into a structured object that can be used to serve the book via a web interface.
"""

import os
import pickle
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

def clean_html_content(soup: BeautifulSoup) -> BeautifulSoup:

    # Remove dangerous/useless tags
    for tag in soup(['script', 'style', 'iframe', 'video', 'nav', 'form', 'button']):
        tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove input tags
    for tag in soup.find_all('input'):
        tag.decompose()

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


def extract_metadata_robust(book_obj) -> BookMetadata:
    """
    Extracts metadata handling both single and list values.
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
        subjects=get_list('subject')
    )


# --- Main Conversion Logic ---

def _flatten_toc(entries: List[TOCEntry]) -> List[TOCEntry]:
    flat: List[TOCEntry] = []
    for e in entries:
        flat.append(e)
        if e.children:
            flat.extend(_flatten_toc(e.children))
    return flat


def _find_anchor_tag(soup: BeautifulSoup, anchor: str):
    if not anchor:
        return None
    tag = soup.find(id=anchor)
    if tag:
        return tag
    # Fallback for name="..."
    return soup.find(attrs={"name": anchor})


def _slice_html_by_anchors(soup: BeautifulSoup, start_anchor: Optional[str], stop_anchors: List[str]) -> str:
    """
    Returns an HTML fragment starting at `start_anchor` (or top of body when None)
    and ending right before the next occurrence of any of `stop_anchors`.
    Best-effort: works well when chapters start at headers or elements with id.
    """
    body = soup.find('body')
    if not body:
        return str(soup)

    # If no start anchor: start at beginning of body
    if not start_anchor:
        start_node = None
    else:
        start_tag = _find_anchor_tag(soup, start_anchor)
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
        t = _find_anchor_tag(soup, a)
        if t:
            stop_tag_candidates.append(t)

    # Helper to compare document order using next_elements
    def comes_before(a_tag, b_tag) -> bool:
        if a_tag is None or b_tag is None:
            return False
        it = a_tag.next_elements
        for el in it:
            if el == b_tag:
                return True
        return False

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

    # Collect HTML between start and stop
    html_parts: List[str] = []
    collecting = False if start_node is not None else True

    for child in body.children:
        # We operate at the top level of body to avoid deep nested slicing issues
        if not collecting and start_node is not None:
            # Start when we reach start_node or it is contained within this child
            if child == start_node or (isinstance(child, Tag) and child.find(id=start_anchor) is not None):
                collecting = True

        if collecting:
            # Stop before stop_tag if it's this child or contained
            if stop_tag is not None:
                if child == stop_tag:
                    break
                if isinstance(child, Tag):
                    # If the stop tag is inside this child, stop here
                    stop_id = stop_tag.get('id') if hasattr(stop_tag, 'get') else None
                    if stop_id and child.find(id=stop_id) is not None:
                        break
                    # As a fallback, check descendants identity
                    try:
                        for d in child.descendants:
                            if d is stop_tag:
                                raise StopIteration
                    except StopIteration:
                        break
            html_parts.append(str(child))

        # If starting at beginning, start collecting immediately
        if start_node is None:
            collecting = True

    # Fallback: if nothing collected, return full body contents
    if not html_parts:
        return "".join(str(x) for x in body.contents)
    return "".join(html_parts)


def process_epub(epub_path: str, output_dir: str) -> Book:

    # 1. Load Book
    print(f"Loading {epub_path}...")
    book = epub.read_epub(epub_path)

    # 2. Extract Metadata
    metadata = extract_metadata_robust(book)

    # 3. Prepare Output Directories
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # 4. Extract Images & Build Map
    print("Extracting images...")
    image_map = {} # Key: internal_path, Value: local_relative_path

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_IMAGE:
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

    # 5. Process TOC
    print("Parsing Table of Contents...")
    toc_structure = parse_toc_recursive(book.toc)
    if not toc_structure:
        print("Warning: Empty TOC, building fallback from Spine...")
        toc_structure = get_fallback_toc(book)
    flat_toc = _flatten_toc(toc_structure)

    # 6. Process Content based on TOC (split by anchors for one-chapter-at-a-time)
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

            # Fix Images
            for img in soup.find_all('img'):
                src = img.get('src', '')
                if not src:
                    continue
                src_decoded = unquote(src)
                filename = os.path.basename(src_decoded)
                if src_decoded in image_map:
                    img['src'] = image_map[src_decoded]
                elif filename in image_map:
                    img['src'] = image_map[filename]

            # Clean HTML
            soup = clean_html_content(soup)
            soups[name] = soup

    # Build per-file anchor lists in TOC order
    anchors_by_file: Dict[str, List[str]] = {}
    for entry in flat_toc:
        file_href = entry.file_href
        anchors_by_file.setdefault(file_href, [])
        if entry.anchor:
            anchors_by_file[file_href].append(entry.anchor)

    # Generate chapters sequentially by TOC order
    for idx, entry in enumerate(flat_toc):
        file_href = entry.file_href
        anchor = entry.anchor or None

        soup = soups.get(file_href)
        if not soup:
            # Missing file; skip
            continue

        # Next anchors in the same file after this one
        anchors_in_file = anchors_by_file.get(file_href, [])
        # Compute the list of stop anchors: anchors that come after this anchor
        stop_anchors: List[str] = []
        if anchor:
            try:
                pos = anchors_in_file.index(anchor)
                stop_anchors = anchors_in_file[pos + 1:]
            except ValueError:
                stop_anchors = anchors_in_file
        else:
            # If starting from top of file, any anchor in file can be a stop
            stop_anchors = anchors_in_file

        fragment_html = _slice_html_by_anchors(soup, anchor, stop_anchors)

        # Extract plain text from the fragment
        frag_soup = BeautifulSoup(fragment_html, 'html.parser')
        text = extract_plain_text(frag_soup)

        ch_href = file_href if not entry.anchor else f"{file_href}#{entry.anchor}"
        chapter = ChapterContent(
            id=f"{file_href}#{entry.anchor}" if entry.anchor else file_href,
            href=ch_href,
            title=entry.title or os.path.splitext(os.path.basename(file_href))[0],
            content=fragment_html,
            text=text,
            order=len(spine_chapters)
        )
        spine_chapters.append(chapter)

    # 7. Final Assembly
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


# --- CLI ---

if __name__ == "__main__":

    import sys
    if len(sys.argv) < 2:
        print("Usage: python reader3.py <file.epub>")
        sys.exit(1)

    epub_file = sys.argv[1]
    assert os.path.exists(epub_file), "File not found."
    out_dir = os.path.splitext(epub_file)[0] + "_data"

    book_obj = process_epub(epub_file, out_dir)
    save_to_pickle(book_obj, out_dir)
    print("\n--- Summary ---")
    print(f"Title: {book_obj.metadata.title}")
    print(f"Authors: {', '.join(book_obj.metadata.authors)}")
    print(f"Physical Files (Spine): {len(book_obj.spine)}")
    print(f"TOC Root Items: {len(book_obj.toc)}")
    print(f"Images extracted: {len(book_obj.images)}")
