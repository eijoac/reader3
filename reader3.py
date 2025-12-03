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

            # Fix Internal Links - rewrite to use a placeholder that will be replaced later
            # We'll use a special format: __INTERNAL_LINK__filename#anchor
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if not href or href.startswith(('http://', 'https://', 'mailto:', 'javascript:', '#')):
                    # External links, email, javascript, or same-page anchors - leave as is
                    continue
                # This is likely an internal epub link
                # Store it in a special format for later processing
                href_decoded = unquote(href)
                link['href'] = f"__INTERNAL_LINK__{href_decoded}"

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

    # Build a mapping from href to spine index for link rewriting
    href_to_index: Dict[str, int] = {}
    
    # Also collect all anchors from all files for comprehensive mapping
    all_file_anchors: Dict[str, List[str]] = {}
    for file_name, soup in soups.items():
        anchors = []
        for tag in soup.find_all(id=True):
            anchors.append(tag['id'])
        for tag in soup.find_all(attrs={"name": True}):
            anchors.append(tag['name'])
        all_file_anchors[file_name] = anchors
    
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
        
        current_index = len(spine_chapters) - 1
        
        # Map both full href and file-only href to this index
        href_to_index[ch_href] = current_index
        href_to_index[file_href] = current_index
        # Also map just the basename
        href_to_index[os.path.basename(file_href)] = current_index
        if entry.anchor:
            # Map the full path with anchor
            href_to_index[f"{file_href}#{entry.anchor}"] = current_index
            # Map the basename with anchor
            href_to_index[os.path.basename(file_href) + '#' + entry.anchor] = current_index
        
        # Map TOC anchors in this file to their exact chapter indices
        # Later we will assign non-TOC anchors to the nearest preceding chapter.
    
    # Assign non-TOC anchors to nearest preceding chapter in the same file
    for file_name, soup in soups.items():
        # Collect chapter starts (by anchor) for this file and resolve their DOM positions
        chapter_starts: List[Tuple[Optional[str], int]] = []  # (anchor, index)
        for i, ch in enumerate(spine_chapters):
            if ch.href.split('#')[0] == file_name:
                ch_anchor = ch.href.split('#')[1] if '#' in ch.href else None
                chapter_starts.append((ch_anchor, i))

        def tag_position(soup: BeautifulSoup, tag: Optional[Tag]) -> int:
            if tag is None:
                return 0
            # Walk the document to get a stable order metric
            pos = 1
            body = soup.find('body') or soup
            for el in body.descendants:
                if el is tag:
                    return pos
                pos += 1
            return pos

        # Resolve tags and sort starts by DOM order
        start_tags: List[Tuple[Optional[str], int, Optional[Tag], int]] = []  # (anchor, idx, tag, pos)
        for anchor_id, idx in chapter_starts:
            if anchor_id is None:
                start_tags.append((None, idx, None, 0))
            else:
                tag = _find_anchor_tag(soup, anchor_id)
                start_tags.append((anchor_id, idx, tag, tag_position(soup, tag)))
        start_tags.sort(key=lambda t: t[3])

        # For each discovered anchor in file, assign chapter index by containment:
        # 1) If exact match to a start anchor, choose that chapter.
        # 2) Else choose the nearest PREVIOUS start (largest s_pos <= a_pos),
        #    which means the anchor lies within that chapter's slice.
        for anchor_id in all_file_anchors.get(file_name, []):
            a_tag = _find_anchor_tag(soup, anchor_id)
            if not a_tag:
                continue
            chosen_idx = None
            a_pos = tag_position(soup, a_tag)
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
    for chapter in spine_chapters:
        # Replace __INTERNAL_LINK__ markers with actual indices
        content = chapter.content
        
        for old_href, chapter_idx in href_to_index.items():
            # Try various formats and preserve fragment if present
            marker = f'__INTERNAL_LINK__{old_href}'
            if marker in content:
                if '#' in old_href:
                    frag = '#' + old_href.split('#', 1)[1]
                else:
                    frag = ''
                content = content.replace(f'href=\"{marker}\"', f'href=\"__CHAPTER__{chapter_idx}{frag}\"')
                content = content.replace(f"href='{marker}'", f"href='__CHAPTER__{chapter_idx}{frag}'")
        
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
    print(f"Cover Image Extracted: {'Yes' if book_obj.metadata.cover_image else 'No'}")
    print(f"Other Images extracted: {len(book_obj.images)}")
