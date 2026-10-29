import json
from pathlib import Path

import pytest
from ebooklib import epub

from converter import convert

ROOT = Path(__file__).resolve().parent.parent


def png(tag: str) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + tag.encode()


HOSTILE_BODY = """
<script>alert(1)</script>
<p onclick="alert(2)">click</p>
<a href="javascript:alert(3)">js</a>
<a href="JavaScript:alert(4)">js upper</a>
<a href="  javascript:alert(5)">js space</a>
<a href="java&#9;script:alert(6)">js tab</a>
<a href="java%73cript:alert(7)">js pct</a>
<a href="data:text/html,<script>alert(8)</script>">data</a>
<a href="https://example.com/ok">ok</a>
<iframe src="https://example.com"></iframe>
<!-- secret comment -->
"""


def build_epub(path: Path) -> Path:
    """
    A small EPUB exercising the converter's tricky paths: a multi-anchor file, a file missing
    from the TOC, colliding image names, an SVG cover wrapper and hostile markup.
    """
    bk = epub.EpubBook()
    bk.set_identifier("test-id")
    bk.set_title("Test Book")
    bk.set_language("en")
    bk.add_author("Test Author")

    images = {
        "img/a/1.png": "A1",
        "img/b/1.png": "B1",
        "img/Pic.png": "UP",
        "img/pic.png": "LOW",
    }
    for name, tag in images.items():
        bk.add_item(epub.EpubItem(uid=name.replace("/", "_"), file_name=name,
                                  media_type="image/png", content=png(tag)))
    bk.add_item(epub.EpubItem(uid="cover_img", file_name="cover.png",
                              media_type="image/png", content=png("COVER")))

    cover = epub.EpubHtml(title="Cover", file_name="cover.xhtml", lang="en")
    cover.content = ('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">'
                     '<image xlink:href="cover.png"/></svg>')

    ch1 = epub.EpubHtml(title="One", file_name="text/ch1.xhtml", lang="en")
    ch1.content = (
        '<h1 id="a">Part A</h1><p>alpha text</p>'
        '<p><img src="../img/a/1.png"/><img src="../img/b/1.png"/>'
        '<img src="../img/Pic.png"/><img src="../img/pic.png"/></p>'
        '<h1 id="b">Part B</h1><p>beta text</p>'
    )
    ch2 = epub.EpubHtml(title="Two", file_name="text/ch2.xhtml", lang="en")
    ch2.content = ('<h1>Two</h1><p>second chapter</p>'
                   '<a href="ch1.xhtml#b">to part b</a> <a href="#local">local</a>'
                   + HOSTILE_BODY)
    extra = epub.EpubHtml(title="Extra", file_name="text/extra.xhtml", lang="en")
    extra.content = "<h1>Not in TOC</h1><p>extra text</p>"

    for item in (cover, ch1, ch2, extra):
        bk.add_item(item)
    bk.toc = [
        epub.Link("text/ch1.xhtml#a", "Part A", "a"),
        epub.Link("text/ch1.xhtml#b", "Part B", "b"),
        epub.Link("text/ch2.xhtml", "Two", "two"),
    ]
    # The spine puts the file the TOC omits before ch2.
    bk.spine = [cover, ch1, extra, ch2]
    bk.add_item(epub.EpubNcx())
    epub.write_epub(str(path), bk)
    return path


@pytest.fixture(scope="session")
def epub_path(tmp_path_factory) -> Path:
    return build_epub(tmp_path_factory.mktemp("epub") / "test.epub")


@pytest.fixture(scope="session")
def converted(tmp_path_factory, epub_path):
    out = tmp_path_factory.mktemp("lib") / "Test_data"
    book = convert.convert_epub(str(epub_path), str(out))
    return book, out


@pytest.fixture
def library(tmp_path, converted, monkeypatch):
    """A library dir holding one converted book, wired into the server module."""
    import shutil

    import server

    _, out = converted
    shutil.copytree(out, tmp_path / "Test_data")
    monkeypatch.setattr(server, "BOOKS_DIR", str(tmp_path))
    monkeypatch.setattr(server, "templates", server.make_templates(ROOT / "templates"))
    server._load_book.cache_clear()
    yield tmp_path
    server._load_book.cache_clear()


@pytest.fixture
def client(library):
    from fastapi.testclient import TestClient

    import server
    return TestClient(server.app)


def load_json(folder: Path) -> dict:
    return json.loads((folder / "book.json").read_text(encoding="utf-8"))
