import os
import pickle
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from reader3 import Book, BookMetadata, ChapterContent, TOCEntry

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Where are the book folders located?
BOOKS_DIR = "library"

@lru_cache(maxsize=10)
def load_book_cached(folder_name: str) -> Optional[Book]:
    """
    Loads the book from the pickle file.
    Cached so we don't re-read the disk on every click.
    """
    file_path = os.path.join(BOOKS_DIR, folder_name, "book.pkl")
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "rb") as f:
            book = pickle.load(f)
        return book
    except Exception as e:
        print(f"Error loading book {folder_name}: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def library_view(request: Request):
    """Lists all available processed books."""
    books = []

    # Scan directory for folders ending in '_data' that have a book.pkl
    if os.path.exists(BOOKS_DIR):
        for item in os.listdir(BOOKS_DIR):
            folder_path = os.path.join(BOOKS_DIR, item)
            if item.endswith("_data") and os.path.isdir(folder_path):
                # Try to load it to get the title
                book = load_book_cached(item)
                if book:
                    books.append({
                        "id": item,
                        "title": book.metadata.title,
                        "author": ", ".join(book.metadata.authors),
                        "chapters": len(book.spine),
                        "has_cover": book.metadata.cover_image is not None
                    })

    return templates.TemplateResponse("library.html", {"request": request, "books": books})

@app.get("/read/{book_id}/images/{image_name}")
async def serve_image(book_id: str, image_name: str):
    """
    Serves images specifically for a book.
    The HTML contains <img src="images/pic.jpg">.
    The browser resolves this to /read/{book_id}/images/pic.jpg.
    """
    # Security check: ensure book_id is clean
    safe_book_id = os.path.basename(book_id)
    safe_image_name = os.path.basename(image_name)

    img_path = os.path.join(BOOKS_DIR, safe_book_id, "images", safe_image_name)

    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(img_path)

@app.get("/read/{book_id}", response_class=HTMLResponse)
async def redirect_to_first_chapter(request: Request, book_id: str):
    """Helper to just go to chapter 0."""
    return await read_chapter(request=request, book_id=book_id, chapter_ref="0")

@app.get("/read/{book_id}/{chapter_ref:path}", response_class=HTMLResponse)
async def read_chapter(request: Request, book_id: str, chapter_ref: str):
    """The main reader interface. Accepts either integer index or filename."""
    book = load_book_cached(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    # Try to parse as integer first
    try:
        chapter_index = int(chapter_ref)
    except ValueError:
        # Not an integer, try to find by href
        chapter_index = None
        for idx, chapter in enumerate(book.spine):
            # Check if the ref matches the chapter's href or id
            if (chapter.href == chapter_ref or 
                chapter.id == chapter_ref or
                chapter.href.endswith(chapter_ref) or
                os.path.basename(chapter.href) == chapter_ref or
                os.path.basename(chapter.href) == os.path.basename(chapter_ref)):
                chapter_index = idx
                break
        
        if chapter_index is None:
            raise HTTPException(status_code=404, detail=f"Chapter not found: {chapter_ref}")

    if chapter_index < 0 or chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter index out of range")

    current_chapter = book.spine[chapter_index]
    
    # Replace __CHAPTER__ markers in content with actual URLs
    content = current_chapter.content
    import re
    # Replace with optional fragment preserved: __CHAPTER__<idx>#fragment
    def repl(m):
        idx = m.group(1)
        frag = m.group(2) or ''
        return f'href="/read/{book_id}/{idx}{frag}"'
    content = re.sub(r'href=["\']__CHAPTER__([0-9]+)(#[^"\']*)?["\']', repl, content)
    
    # Create a modified chapter object with rewritten content
    class ModifiedChapter:
        def __init__(self, original, new_content):
            self.id = original.id
            self.href = original.href
            self.title = original.title
            self.content = new_content
            self.text = original.text
            self.order = original.order
    
    modified_chapter = ModifiedChapter(current_chapter, content)

    # Calculate Prev/Next links
    prev_idx = chapter_index - 1 if chapter_index > 0 else None
    next_idx = chapter_index + 1 if chapter_index < len(book.spine) - 1 else None

    return templates.TemplateResponse("reader.html", {
        "request": request,
        "book": book,
        "current_chapter": modified_chapter,
        "chapter_index": chapter_index,
        "book_id": book_id,
        "prev_idx": prev_idx,
        "next_idx": next_idx
    })

@app.get("/cover/{book_id}")
async def serve_cover(book_id: str):
    """
    Serves the cover image for a book in the library view.
    """
    book = load_book_cached(book_id)
    if not book or not book.metadata.cover_image:
        raise HTTPException(status_code=404, detail="Cover not found")
    
    # Security check: ensure book_id is clean
    safe_book_id = os.path.basename(book_id)
    cover_filename = os.path.basename(book.metadata.cover_image)
    
    cover_path = os.path.join(BOOKS_DIR, safe_book_id, "images", cover_filename)
    
    if not os.path.exists(cover_path):
        raise HTTPException(status_code=404, detail="Cover image file not found")
    
    return FileResponse(cover_path)

@app.get("/read/{book_id}/images/{image_name}")
async def serve_image(book_id: str, image_name: str):
    """
    Serves images specifically for a book.
    The HTML contains <img src="images/pic.jpg">.
    The browser resolves this to /read/{book_id}/images/pic.jpg.
    """
    # Security check: ensure book_id is clean
    safe_book_id = os.path.basename(book_id)
    safe_image_name = os.path.basename(image_name)

    img_path = os.path.join(BOOKS_DIR, safe_book_id, "images", safe_image_name)

    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(img_path)

if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://127.0.0.1:8123")
    uvicorn.run(app, host="127.0.0.1", port=8123)
