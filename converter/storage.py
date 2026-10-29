"""Reading and writing a converted book: book.json (everything) and meta.json (a small summary)."""

import json
import os
from dataclasses import asdict

from .models import Book, BookMetadata, ChapterContent, TOCEntry

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
    with open(os.path.join(book_dir, BOOK_FILE), encoding="utf-8") as f:
        return book_from_dict(json.load(f))


def read_summary(book_dir: str) -> dict | None:
    """The summary from meta.json, or None if it is missing, unreadable or from another format version."""
    try:
        with open(os.path.join(book_dir, META_FILE), encoding="utf-8") as f:
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
