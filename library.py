"""The library on disk: which books exist, loading them, and caching what is derived from them."""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

from converter.models import Book, TOCEntry
from converter.storage import BOOK_FILE, is_legacy_book, load_book_file, read_summary

log = logging.getLogger("uvicorn.error")  # shares uvicorn's handlers, so messages show up in the console

# Where are the book folders located?
BOOKS_DIR = "library"


def is_valid_book_id(book_id: str) -> bool:
    """A book id is the name of a processed folder: a plain `*_data` name, never a path."""
    return (
        book_id.endswith("_data")
        and book_id == os.path.basename(book_id)
        and "\\" not in book_id
        and "\x00" not in book_id
    )


def book_dir(book_id: str) -> str | None:
    """The folder of a book, or None if the id is not a valid, existing book folder."""
    if not is_valid_book_id(book_id):
        return None
    path = os.path.join(BOOKS_DIR, book_id)
    return path if os.path.isdir(path) else None


def is_outdated(book_id: str) -> bool:
    """True for books converted by an older version (they need `reprocess`)."""
    return is_valid_book_id(book_id) and is_legacy_book(os.path.join(BOOKS_DIR, book_id))


def list_books() -> list[dict]:
    """Summaries for the library page, read from each book's small meta.json (never the whole book)."""
    books = []
    if not os.path.isdir(BOOKS_DIR):
        return books
    for item in sorted(os.listdir(BOOKS_DIR)):
        folder_path = os.path.join(BOOKS_DIR, item)
        if not (item.endswith("_data") and os.path.isdir(folder_path)):
            continue
        meta = read_summary(folder_path)
        if meta:
            books.append(
                {
                    "id": item,
                    "title": meta["title"],
                    "author": ", ".join(meta.get("authors", [])),
                    "chapters": meta.get("chapters", 0),
                    "has_cover": bool(meta.get("cover_image")),
                    "outdated": False,
                }
            )
        elif is_legacy_book(folder_path):
            books.append(
                {
                    "id": item,
                    "title": item[: -len("_data")],
                    "author": "",
                    "chapters": 0,
                    "has_cover": False,
                    "outdated": True,
                }
            )
    return books


@dataclass(frozen=True)
class LoadedBook:
    """A book plus everything derived from it that is the same on every request."""

    book: Book
    toc_index: dict[str, int]  # TOC href -> chapter index; read-only, shared between requests


@lru_cache(maxsize=16)
def _load_book(folder_name: str, mtime_ns: int) -> LoadedBook | None:
    # mtime_ns is unused in the body; it is part of the cache key so a reprocessed book is reloaded
    try:
        book = load_book_file(os.path.join(BOOKS_DIR, folder_name))
    except Exception:
        log.exception("Error loading book %s", folder_name)
        return None
    return LoadedBook(book, toc_index_map(book))


def load_loaded_book(folder_name: str) -> LoadedBook | None:
    """
    Loads the book from book.json together with its derived data.
    Cached so we don't re-read or re-derive anything on every click, but the cache is
    invalidated whenever book.json changes (e.g. the book is reprocessed).
    """
    if not is_valid_book_id(folder_name):
        return None
    file_path = os.path.join(BOOKS_DIR, folder_name, BOOK_FILE)
    try:
        mtime_ns = os.stat(file_path).st_mtime_ns
    except OSError:
        return None
    return _load_book(folder_name, mtime_ns)


def load_book_cached(folder_name: str) -> Book | None:
    loaded = load_loaded_book(folder_name)
    return loaded.book if loaded else None


def toc_index_map(book: Book) -> dict[str, int]:
    """
    Maps every TOC href to the index of the chapter it opens: the exact chapter
    (file#anchor) if there is one, otherwise the first chapter of that file.
    """
    exact: dict[str, int] = {}
    first_in_file: dict[str, int] = {}
    for idx, chapter in enumerate(book.spine):
        exact.setdefault(chapter.href, idx)
        first_in_file.setdefault(chapter.href.split("#")[0], idx)

    result: dict[str, int] = {}

    def walk(entries: list[TOCEntry]):
        for entry in entries:
            idx = exact.get(entry.href, first_in_file.get(entry.file_href))
            if idx is not None:
                result[entry.href] = idx
            walk(entry.children)

    walk(book.toc)
    return result
