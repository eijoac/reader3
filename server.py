import logging
import os
import posixpath
import re
from dataclasses import replace
from functools import lru_cache
from typing import Dict, List, Optional
from urllib.parse import quote

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from reader3 import (
    BOOK_FILE, Book, TOCEntry, is_legacy_book, load_book_file, read_summary,
)

log = logging.getLogger("uvicorn.error")  # shares uvicorn's handlers, so messages show up in the console

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Where are the book folders located?
BOOKS_DIR = "library"

# Placeholder written by reader3.py: __CHAPTER__<spine index>[#fragment]
CHAPTER_LINK_RE = re.compile(r'href=["\']__CHAPTER__([0-9]+)(#[^"\']*)?["\']')


def _is_valid_book_id(book_id: str) -> bool:
    """A book id is the name of a processed folder: a plain `*_data` name, never a path."""
    return (
        book_id.endswith("_data")
        and book_id == os.path.basename(book_id)
        and "\\" not in book_id
        and "\x00" not in book_id
    )


def _book_dir(book_id: str) -> str:
    """Resolves a book id to its folder, or raises 404. Every route goes through this."""
    if not _is_valid_book_id(book_id):
        raise HTTPException(status_code=404, detail="Book not found")
    path = os.path.join(BOOKS_DIR, book_id)
    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail="Book not found")
    return path


def _raise_if_outdated(book_id: str):
    if is_legacy_book(os.path.join(BOOKS_DIR, book_id)):
        name = book_id[:-len("_data")]
        raise HTTPException(
            status_code=409,
            detail=f"This book was converted by an older version. Run: uv run reader3.py reprocess {name}",
        )


@lru_cache(maxsize=16)
def _load_book(folder_name: str, mtime_ns: int) -> Optional[Book]:
    # mtime_ns is unused in the body; it is part of the cache key so a reprocessed book is reloaded
    try:
        return load_book_file(os.path.join(BOOKS_DIR, folder_name))
    except Exception:
        log.exception("Error loading book %s", folder_name)
        return None


def load_book_cached(folder_name: str) -> Optional[Book]:
    """
    Loads the book from book.json.
    Cached so we don't re-read the disk on every click, but the cache is
    invalidated whenever book.json changes (e.g. the book is reprocessed).
    """
    if not _is_valid_book_id(folder_name):
        return None
    file_path = os.path.join(BOOKS_DIR, folder_name, BOOK_FILE)
    try:
        mtime_ns = os.stat(file_path).st_mtime_ns
    except OSError:
        return None
    return _load_book(folder_name, mtime_ns)


def _toc_index_map(book: Book) -> Dict[str, int]:
    """
    Maps every TOC href to the index of the chapter it opens: the exact chapter
    (file#anchor) if there is one, otherwise the first chapter of that file.
    """
    exact: Dict[str, int] = {}
    first_in_file: Dict[str, int] = {}
    for idx, chapter in enumerate(book.spine):
        exact.setdefault(chapter.href, idx)
        first_in_file.setdefault(chapter.href.split('#')[0], idx)

    result: Dict[str, int] = {}

    def walk(entries: List[TOCEntry]):
        for entry in entries:
            idx = exact.get(entry.href, first_in_file.get(entry.file_href))
            if idx is not None:
                result[entry.href] = idx
            walk(entry.children)

    walk(book.toc)
    return result

@app.get("/", response_class=HTMLResponse)
async def library_view(request: Request):
    """Lists all available processed books."""
    books = []

    # Folders ending in '_data'; the title etc. come from the small meta.json, not the whole book
    if os.path.isdir(BOOKS_DIR):
        for item in sorted(os.listdir(BOOKS_DIR)):
            folder_path = os.path.join(BOOKS_DIR, item)
            if not (item.endswith("_data") and os.path.isdir(folder_path)):
                continue
            meta = read_summary(folder_path)
            if meta:
                books.append({
                    "id": item,
                    "title": meta["title"],
                    "author": ", ".join(meta.get("authors", [])),
                    "chapters": meta.get("chapters", 0),
                    "has_cover": bool(meta.get("cover_image")),
                    "outdated": False,
                })
            elif is_legacy_book(folder_path):
                books.append({
                    "id": item, "title": item[:-len("_data")], "author": "", "chapters": 0,
                    "has_cover": False, "outdated": True,
                })

    return templates.TemplateResponse(request, "library.html", {"books": books})

@app.get("/read/{book_id}/images/{image_name}")
async def serve_image(book_id: str, image_name: str):
    """
    Serves images specifically for a book.
    The HTML contains <img src="images/pic.jpg">.
    The browser resolves this to /read/{book_id}/images/pic.jpg.
    """
    # Security check: the id must be a plain book folder name
    img_path = os.path.join(_book_dir(book_id), "images", os.path.basename(image_name))

    if not os.path.isfile(img_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(img_path)

@app.get("/read/{book_id}")
async def redirect_to_first_chapter(book_id: str):
    """Canonical URL for a book is its first chapter."""
    _book_dir(book_id)
    return RedirectResponse(f"/read/{quote(book_id, safe='')}/0")


def _resolve_chapter(book: Book, ref: str) -> Optional[int]:
    """
    Maps a URL reference to a spine index. Accepts a numeric index, or an exact
    chapter href / id / file name (the first chapter of that file). Returns None if
    nothing matches; a numeric ref is range-checked by the caller.
    """
    if ref.isascii() and ref.isdigit():
        return int(ref)
    for idx, chapter in enumerate(book.spine):
        if ref in (chapter.href, chapter.id):
            return idx
    for idx, chapter in enumerate(book.spine):
        file_href = chapter.href.split('#')[0]
        if ref in (file_href, posixpath.basename(file_href)):
            return idx
    return None


@app.get("/read/{book_id}/{chapter_ref:path}", response_class=HTMLResponse)
async def read_chapter(request: Request, book_id: str, chapter_ref: str):
    """The main reader interface. Accepts either integer index or filename."""
    book = load_book_cached(book_id)
    if not book:
        _raise_if_outdated(book_id)
        raise HTTPException(status_code=404, detail="Book not found")

    chapter_index = _resolve_chapter(book, chapter_ref)
    if chapter_index is None:
        raise HTTPException(status_code=404, detail=f"Chapter not found: {chapter_ref}")
    if chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter index out of range")

    book_url = quote(book_id, safe='')

    # Replace __CHAPTER__ markers in content with actual URLs (optional fragment preserved)
    def repl(m):
        frag = m.group(2) or ''
        return f'href="/read/{book_url}/{m.group(1)}{frag}"'
    content = CHAPTER_LINK_RE.sub(repl, book.spine[chapter_index].content)
    current_chapter = replace(book.spine[chapter_index], content=content)

    # Calculate Prev/Next links
    prev_idx = chapter_index - 1 if chapter_index > 0 else None
    next_idx = chapter_index + 1 if chapter_index < len(book.spine) - 1 else None

    return templates.TemplateResponse(request, "reader.html", {
        "book": book,
        "current_chapter": current_chapter,
        "chapter_index": chapter_index,
        "book_id": book_id,
        "book_url": book_url,
        "toc_index": _toc_index_map(book),
        "prev_idx": prev_idx,
        "next_idx": next_idx
    })

@app.get("/cover/{book_id}")
async def serve_cover(book_id: str):
    """
    Serves the cover image for a book in the library view.
    """
    book_dir = _book_dir(book_id)  # also rejects anything that isn't a plain book folder name
    meta = read_summary(book_dir)
    if not meta or not meta.get("cover_image"):
        raise HTTPException(status_code=404, detail="Cover not found")

    cover_path = os.path.join(book_dir, "images", os.path.basename(meta["cover_image"]))

    if not os.path.isfile(cover_path):
        raise HTTPException(status_code=404, detail="Cover image file not found")

    return FileResponse(cover_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8123)
