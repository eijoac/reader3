import os
import posixpath
import re
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import library
from converter.models import Book
from converter.storage import read_summary

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
# Book HTML is untrusted. Even if the sanitizer missed something, the browser must not run it:
# scripts only from our own /static, no plugins, no framing, no remote images or requests.
# Inline styles stay allowed because books carry style attributes.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; font-src 'self' data:; connect-src 'self'; "
        "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/") and response.status_code in (200, 304):
        # Pages link assets as ?v=<mtime> (see static_url), so a versioned URL never goes stale
        response.headers["Cache-Control"] = (
            "public, max-age=31536000, immutable" if "v" in request.query_params else "no-cache"
        )
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    return response


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def static_url(name: str) -> str:
    """URL of a static file, versioned by its modification time so browsers can cache it indefinitely."""
    return f"/static/{name}?v={(BASE_DIR / 'static' / name).stat().st_mtime_ns}"


def make_templates(directory) -> Jinja2Templates:
    result = Jinja2Templates(directory=directory)
    result.env.globals["static_url"] = static_url
    return result


templates = make_templates(BASE_DIR / "templates")

# Placeholder written by reader3.py: __CHAPTER__<spine index>[#fragment]
CHAPTER_LINK_RE = re.compile(r'href=["\']__CHAPTER__([0-9]+)(#[^"\']*)?["\']')
MAX_SEARCH_RESULTS = 50
SEARCH_CONTEXT_CHARS = 80


def _book_dir(book_id: str) -> str:
    """Resolves a book id to its folder, or raises 404. Every route goes through this."""
    path = library.book_dir(book_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Book not found")
    return path


def _raise_if_outdated(book_id: str):
    if library.is_outdated(book_id):
        name = book_id[: -len("_data")]
        raise HTTPException(
            status_code=409,
            detail=f"This book was converted by an older version. Run: uv run reader3.py reprocess {name}",
        )


@app.get("/", response_class=HTMLResponse)
def library_view(request: Request):
    """Lists all available processed books."""
    return templates.TemplateResponse(request, "library.html", {"books": library.list_books()})


@app.get("/read/{book_id}/images/{image_name}")
def serve_image(book_id: str, image_name: str):
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
def redirect_to_first_chapter(book_id: str):
    """Canonical URL for a book is its first chapter."""
    _book_dir(book_id)
    return RedirectResponse(f"/read/{quote(book_id, safe='')}/0")


@app.get("/read/{book_id}/search")
def search_book(book_id: str, q: str = Query(min_length=1, max_length=120)):
    """Search chapter text and return a bounded set of plain-text snippets."""
    loaded = library.load_loaded_book(book_id)
    if not loaded:
        _raise_if_outdated(book_id)
        raise HTTPException(status_code=404, detail="Book not found")

    query = q.strip()
    if not query:
        return {"results": [], "truncated": False}

    results = []
    truncated = False
    for chapter_index, chapter in enumerate(loaded.book.spine):
        match = re.search(re.escape(query), chapter.text, re.IGNORECASE)
        if match is None:
            continue
        if len(results) == MAX_SEARCH_RESULTS:
            truncated = True
            break
        start = max(0, match.start() - SEARCH_CONTEXT_CHARS)
        end = min(len(chapter.text), match.end() + SEARCH_CONTEXT_CHARS)
        snippet = chapter.text[start:end]
        if start:
            snippet = "..." + snippet
        if end < len(chapter.text):
            snippet += "..."
        results.append(
            {
                "chapter": chapter_index,
                "title": chapter.title,
                "snippet": snippet,
            }
        )

    return {"results": results, "truncated": truncated}


def _resolve_chapter(book: Book, ref: str) -> int | None:
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
        file_href = chapter.href.split("#")[0]
        if ref in (file_href, posixpath.basename(file_href)):
            return idx
    return None


@app.get("/read/{book_id}/{chapter_ref:path}", response_class=HTMLResponse)
def read_chapter(request: Request, book_id: str, chapter_ref: str):
    """The main reader interface. Accepts either integer index or filename."""
    loaded = library.load_loaded_book(book_id)
    if not loaded:
        _raise_if_outdated(book_id)
        raise HTTPException(status_code=404, detail="Book not found")
    book = loaded.book

    chapter_index = _resolve_chapter(book, chapter_ref)
    if chapter_index is None:
        raise HTTPException(status_code=404, detail=f"Chapter not found: {chapter_ref}")
    if chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter index out of range")

    book_url = quote(book_id, safe="")

    # Replace __CHAPTER__ markers in content with actual URLs (optional fragment preserved)
    def repl(m):
        frag = m.group(2) or ""
        return f'href="/read/{book_url}/{m.group(1)}{frag}"'

    content = CHAPTER_LINK_RE.sub(repl, book.spine[chapter_index].content)
    current_chapter = replace(book.spine[chapter_index], content=content)

    # Calculate Prev/Next links
    prev_idx = chapter_index - 1 if chapter_index > 0 else None
    next_idx = chapter_index + 1 if chapter_index < len(book.spine) - 1 else None
    toc_targets = set(loaded.toc_index.values())
    front_matter_links = []
    if book.spine:
        front_matter_links.append({"index": 0, "title": "Cover"})
    if len(book.spine) > 1 and 1 not in toc_targets:
        front_matter_links.append({"index": 1, "title": book.spine[1].title})

    return templates.TemplateResponse(
        request,
        "reader.html",
        {
            "book": book,
            "current_chapter": current_chapter,
            "chapter_index": chapter_index,
            "book_id": book_id,
            "book_url": book_url,
            "toc_index": loaded.toc_index,
            "front_matter_links": front_matter_links,
            "prev_idx": prev_idx,
            "next_idx": next_idx,
        },
    )


@app.get("/cover/{book_id}")
def serve_cover(book_id: str):
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
