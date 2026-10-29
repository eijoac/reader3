"""Command line interface: convert, list, reprocess, remove."""

import logging
import os
import shutil
import sys
import tomllib
from pathlib import Path

from .convert import convert_epub
from .models import Book
from .storage import BOOK_FILE, is_legacy_book, read_summary

log = logging.getLogger(__name__)


LIBRARY_DIR = "library"
USAGE = """Usage:
  reader3.py <name.epub>          convert library/<name.epub> (the file must be in library/)
  reader3.py list                 list converted books
  reader3.py reprocess <name>|--all   re-convert from library/<name>.epub
  reader3.py remove <name>        delete a converted book (the .epub is kept)
  reader3.py --version            show the version

Books are converted to library/<name>_data."""


def _book_names(library_dir: str) -> list[str]:
    if not os.path.isdir(library_dir):
        return []
    return sorted(
        d[: -len("_data")]
        for d in os.listdir(library_dir)
        if d.endswith("_data")
        and (os.path.isfile(os.path.join(library_dir, d, BOOK_FILE)) or is_legacy_book(os.path.join(library_dir, d)))
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


def _cmd_reprocess(args: list[str], library_dir: str) -> int:
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


def _cmd_remove(args: list[str], library_dir: str) -> int:
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


def _find_in_library(arg: str, library_dir: str) -> str | None:
    """Accepts `name.epub` or `library/name.epub`; the file must sit directly in the library folder."""
    for candidate in (arg, os.path.join(library_dir, arg)):
        if os.path.isfile(candidate) and os.path.samefile(os.path.dirname(os.path.abspath(candidate)), library_dir):
            return candidate
    return None


def _version() -> str:
    """Read from pyproject.toml (this is run as a plain script, so the project is not installed)."""
    try:
        with open(Path(__file__).resolve().parent.parent / "pyproject.toml", "rb") as f:
            return tomllib.load(f)["project"]["version"]
    except OSError, KeyError, tomllib.TOMLDecodeError:
        return "unknown"


def main(argv: list[str], library_dir: str = LIBRARY_DIR) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = argv[1:]
    if not args:
        print(USAGE, file=sys.stderr)
        return 1
    command, rest = args[0], args[1:]
    if command in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if command in ("-V", "--version"):
        print(f"reader3 {_version()}")
        return 0
    if command.startswith("-"):
        print(f"Error: unknown option: {command}\n\n{USAGE}", file=sys.stderr)
        return 1
    if command == "list":
        return _cmd_list(library_dir)
    if command == "reprocess":
        return _cmd_reprocess(rest, library_dir)
    if command == "remove":
        return _cmd_remove(rest, library_dir)

    epub = _find_in_library(command, library_dir)
    if epub is None:
        print(f"Error: {command} not found. Put the .epub in {library_dir}/ first.", file=sys.stderr)
        return 1
    return 0 if _convert_cli(epub, library_dir) else 1
