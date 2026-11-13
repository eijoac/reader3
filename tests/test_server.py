import json
import os
import re

import pytest

import library as libmod
import server
from converter import storage


def test_library_lists_book_with_encoded_links(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Test Book" in r.text
    assert 'href="/read/Test_data/0"' in r.text


def test_book_url_redirects_to_first_chapter(client):
    r = client.get("/read/Test_data", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"].endswith("/read/Test_data/0")


def test_chapter_pages_render_for_every_index(client, converted):
    book, _ = converted
    for i in range(len(book.spine)):
        assert client.get(f"/read/Test_data/{i}").status_code == 200


@pytest.mark.parametrize("ref", ["999", "-1", "abc", "1.5", "%201", "xtra.xhtml", "ch1.xhtml%23"])
def test_bad_chapter_refs_are_404(client, ref):
    assert client.get(f"/read/Test_data/{ref}").status_code == 404


def test_chapter_ref_by_href_and_file_name(client, converted):
    book, _ = converted
    assert client.get(f"/read/Test_data/{book.spine[2].href}").status_code == 200
    assert client.get("/read/Test_data/text/extra.xhtml").status_code == 200


def test_toc_links_are_real_and_active_is_marked(client):
    html = client.get("/read/Test_data/2").text
    assert 'href="#"' not in html and "findAndGo" not in html
    active = re.findall(r'<a[^>]*class="toc-link active"[^>]*>', html)
    assert len(active) == 1 and 'href="/read/Test_data/2"' in active[0]
    assert "aria-current" in active[0]


def test_toc_starts_with_cover_and_unlinked_second_section(client, library):
    book_path = library / "Test_data" / "book.json"
    data = json.loads(book_path.read_text())
    data["toc"] = data["toc"][1:]
    book_path.write_text(json.dumps(data))

    html = client.get("/read/Test_data/0").text
    cover = html.index('<a href="/read/Test_data/0"')
    second_section = html.index('<a href="/read/Test_data/1"')
    regular_toc = html.index('<a href="/read/Test_data/2"')
    assert cover < second_section < regular_toc
    assert "Cover" in html[cover : html.index("</a>", cover)]
    assert "Part A" in html[second_section : html.index("</a>", second_section)]


def test_chapter_links_are_rewritten(client, converted):
    book, _ = converted
    two_index = next(i for i, c in enumerate(book.spine) if c.title == "Two")
    html = client.get(f"/read/Test_data/{two_index}").text
    assert "__CHAPTER__" not in html
    assert re.search(r'href="/read/Test_data/\d+#b"', html)


def test_book_search_returns_case_insensitive_section_snippets(client, converted):
    book, _ = converted
    expected_index = next(i for i, chapter in enumerate(book.spine) if "alpha text" in chapter.text)
    response = client.get("/read/Test_data/search", params={"q": "ALPHA TEXT"})
    assert response.status_code == 200
    data = response.json()
    assert data["truncated"] is False
    assert len(data["results"]) == 1
    result = data["results"][0]
    assert result["chapter"] == expected_index
    assert result["title"] == book.spine[expected_index].title
    assert "alpha text" in result["snippet"].lower()


def test_book_search_handles_blank_and_invalid_queries(client):
    assert client.get("/read/Test_data/search", params={"q": "   "}).json() == {
        "results": [],
        "truncated": False,
    }
    assert client.get("/read/Test_data/search", params={"q": "x" * 121}).status_code == 422
    assert client.get("/read/Missing_data/search", params={"q": "alpha"}).status_code == 404


def test_book_search_caps_results(client, monkeypatch):
    from types import SimpleNamespace

    chapter = SimpleNamespace(title="Match", text="needle")
    loaded = SimpleNamespace(book=SimpleNamespace(spine=[chapter] * (server.MAX_SEARCH_RESULTS + 1)))
    monkeypatch.setattr(libmod, "load_loaded_book", lambda book_id: loaded)

    data = client.get("/read/Test_data/search", params={"q": "needle"}).json()
    assert len(data["results"]) == server.MAX_SEARCH_RESULTS
    assert data["truncated"] is True


def test_images_and_cover_served(client, converted):
    book, _ = converted
    assert client.get("/cover/Test_data").status_code == 200
    image = book.metadata.cover_image.split("/", 1)[1]
    r = client.get(f"/read/Test_data/images/{image}")
    assert r.status_code == 200 and r.content.startswith(b"\x89PNG")
    assert client.get("/read/Test_data/images/missing.png").status_code == 404
    assert client.get("/read/Test_data/images/..%2Fbook.json").status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        "/read/..%2F..%2Fetc/0",
        "/read/%2e%2e/0",
        "/read/library/0",
        "/read/Nope_data/0",
        "/cover/%2e%2e",
        "/cover/Nope_data",
        "/read/Nope_data",
        "/read/Test%00_data/0",
    ],
)
def test_invalid_book_ids_are_404(client, path):
    assert client.get(path, follow_redirects=False).status_code == 404


def test_folder_without_data_suffix_is_not_served(client, library):
    (library / "plain").mkdir()
    for name in ("book.json", "meta.json"):
        (library / "plain" / name).write_bytes((library / "Test_data" / name).read_bytes())
    assert client.get("/read/plain/0").status_code == 404
    assert client.get("/cover/plain").status_code == 404


def test_is_valid_book_id():
    assert libmod.is_valid_book_id("Book_data")
    assert libmod.is_valid_book_id("My Book #1_data")
    for bad in ("", "Book", "../x_data", "a/b_data", "a\\b_data", "x\x00_data", ".."):
        assert not libmod.is_valid_book_id(bad)


def test_corrupt_book_is_skipped(client, library):
    (library / "Bad_data").mkdir()
    (library / "Bad_data" / "book.json").write_text("garbage")
    (library / "Bad_data" / "meta.json").write_text("garbage")
    r = client.get("/")
    assert r.status_code == 200 and "Test Book" in r.text
    assert client.get("/read/Bad_data/0").status_code == 404


def test_legacy_pickle_book_is_flagged_not_loaded(client, library):
    (library / "Old_data").mkdir()
    (library / "Old_data" / "book.pkl").write_bytes(b"\x80\x04N.")
    page = client.get("/")
    assert page.status_code == 200
    assert "reprocess Old" in page.text and "Test Book" in page.text
    r = client.get("/read/Old_data/0")
    assert r.status_code == 409 and "reprocess Old" in r.text


def test_other_format_version_is_not_loaded(client, library):
    for name in ("book.json", "meta.json"):
        path = library / "Test_data" / name
        data = json.loads(path.read_text())
        data["format_version"] = 999
        path.write_text(json.dumps(data))
    assert client.get("/read/Test_data/0").status_code == 404
    assert "Test Book" not in client.get("/").text


def test_library_page_does_not_load_books(client, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("library page must only read meta.json")

    monkeypatch.setattr(libmod, "load_book_file", boom)
    r = client.get("/")
    assert "Test Book" in r.text
    assert client.get("/cover/Test_data").status_code == 200


def test_roundtrip_through_json_is_lossless(converted):
    book, out = converted
    assert storage.load_book_file(str(out)) == book


def _reader_config(html: str) -> dict:
    m = re.search(r'<script type="application/json" id="reader-config">(.*?)</script>', html, re.S)
    return json.loads(m.group(1))


def test_static_assets_are_served(client):
    for path, marker in [
        ("theme.js", "toggleTheme"),
        ("reader.js", "reader-config"),
        ("reader.css", "--font-scale"),
        ("library.js", "read-btn"),
        ("library.css", "book-grid"),
    ]:
        r = client.get(f"/static/{path}")
        assert r.status_code == 200 and marker in r.text, path
    assert client.get("/static/../server.py").status_code == 404


@pytest.mark.parametrize("url", ["/", "/read/Test_data/0"])
def test_pages_work_under_the_csp(client, url):
    # The CSP only allows scripts from /static: no executable inline scripts, no inline handlers
    html = client.get(url).text
    assert "<style" not in html
    for m in re.finditer(r"<script\b([^>]*)>", html):
        attrs = m.group(1)
        assert "src=" in attrs or "application/json" in attrs, m.group(0)
    assert not re.search(r"\son[a-z]+\s*=", html)
    assert "javascript:" not in html


@pytest.mark.parametrize("url", ["/", "/read/Test_data/0", "/cover/Test_data", "/static/theme.js", "/nope"])
def test_security_headers_on_every_response(client, url):
    r = client.get(url)
    csp = r.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "object-src 'none'" in csp and "frame-ancestors 'none'" in csp
    assert "'unsafe-inline'" not in csp.split("style-src")[0]  # never for scripts
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"


def test_routes_are_threadpool_functions():
    # async routes would run blocking file/JSON work on the event loop
    import inspect

    for route in server.app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint and getattr(route, "path", "").startswith(("/read", "/cover")) or getattr(route, "path", "") == "/":
            assert not inspect.iscoroutinefunction(endpoint), route.path


def test_reader_has_theme_keyboard_and_position_hooks(client):
    html = client.get("/read/Test_data/2").text
    assert 'id="theme-toggle"' in html
    config = _reader_config(html)
    assert config == {
        "bookId": "Test_data",
        "chapter": 2,
        "prevUrl": "/read/Test_data/1",
        "nextUrl": "/read/Test_data/3",
        "searchUrl": "/read/Test_data/search",
    }
    assert "/static/reader.js" in html and "/static/theme.js" in html


def test_reader_controls_have_accessible_semantics(client):
    html = client.get("/read/Test_data/2").text
    assert '<nav id="sidebar" aria-label="Table of contents">' in html
    assert 'aria-controls="sidebar" aria-expanded="true"' in html
    assert '<main id="main" tabindex="-1" aria-label="Book content">' in html
    assert '<label for="book-search">Search this book</label>' in html
    assert 'id="search-status" class="search-status" role="status" aria-live="polite"' in html
    assert '<nav class="chapter-nav" aria-label="Chapter navigation">' in html


def test_library_uses_landmarks_and_book_articles(client):
    html = client.get("/").text
    assert '<header class="page-header">' in html
    assert "<main>" in html and '<section class="book-grid" aria-label="Books">' in html
    assert '<article class="book-card">' in html and '<h2 class="book-title">Test Book</h2>' in html


def test_reader_has_font_controls_and_book_language(client):
    html = client.get("/read/Test_data/2").text
    for control in ("font-smaller", "font-larger", "font-toggle"):
        assert f'id="{control}"' in html
    assert '<html lang="en">' in html


def test_reader_language_comes_from_the_book_and_is_escaped(client, library):
    path = library / "Test_data" / "book.json"
    data = json.loads(path.read_text())
    data["metadata"]["language"] = 'fr"><script>x</script>'
    path.write_text(json.dumps(data))
    html = client.get("/read/Test_data/0").text
    assert "<script>x</script>" not in html


def test_reader_keyboard_urls_are_null_at_the_ends(client, converted):
    book, _ = converted
    assert _reader_config(client.get("/read/Test_data/0").text)["prevUrl"] is None
    last = client.get(f"/read/Test_data/{len(book.spine) - 1}").text
    assert _reader_config(last)["nextUrl"] is None


def test_library_button_carries_data_for_resume(client):
    html = client.get("/").text
    assert 'data-book-id="Test_data"' in html and 'data-chapters="' in html


def test_cache_reloads_when_book_is_reprocessed(library):
    first = libmod.load_book_cached("Test_data")
    assert libmod.load_book_cached("Test_data") is first

    path = library / "Test_data" / "book.json"
    data = json.loads(path.read_text())
    data["metadata"]["title"] = "Changed Title"
    path.write_text(json.dumps(data))
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    assert libmod.load_book_cached("Test_data").metadata.title == "Changed Title"


def test_toc_index_is_computed_once_per_book_version(library, monkeypatch):
    calls = []
    real = libmod.toc_index_map
    monkeypatch.setattr(libmod, "toc_index_map", lambda book: calls.append(1) or real(book))
    libmod._load_book.cache_clear()
    first = libmod.load_loaded_book("Test_data")
    for _ in range(3):
        assert libmod.load_loaded_book("Test_data") is first
    assert first.toc_index and len(calls) == 1

    path = library / "Test_data" / "book.json"
    path.write_text(path.read_text())
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    libmod.load_loaded_book("Test_data")
    assert len(calls) == 2  # recomputed after the book was reprocessed


def test_reader_does_not_rebuild_toc_map_per_request(client, monkeypatch):
    client.get("/read/Test_data/0")  # warm the cache

    def boom(book):
        raise AssertionError("toc map must come from the cache")

    monkeypatch.setattr(libmod, "toc_index_map", boom)
    assert client.get("/read/Test_data/1").status_code == 200


def test_reader_omits_lang_when_the_book_has_none(client, library):
    path = library / "Test_data" / "book.json"
    data = json.loads(path.read_text())
    data["metadata"]["language"] = ""
    path.write_text(json.dumps(data))
    html = client.get("/read/Test_data/0").text
    assert "<html>" in html and "lang=" not in html.split("</head>")[0]


def test_static_urls_are_versioned_and_cached(client):
    html = client.get("/read/Test_data/0").text
    urls = re.findall(r'(?:src|href)="(/static/[^"]+)"', html)
    assert urls and all(re.search(r"\?v=\d+$", u) for u in urls)

    versioned = client.get(urls[0])
    assert versioned.status_code == 200
    assert "immutable" in versioned.headers["cache-control"]
    assert "max-age=31536000" in versioned.headers["cache-control"]
    assert client.get("/static/theme.js").headers["cache-control"] == "no-cache"
    assert "cache-control" not in client.get("/cover/Test_data").headers


def test_static_version_changes_with_the_file(tmp_path, monkeypatch):
    (tmp_path / "static").mkdir()
    f = tmp_path / "static" / "a.js"
    f.write_text("1")
    monkeypatch.setattr(server, "BASE_DIR", tmp_path)
    first = server.static_url("a.js")
    os.utime(f, ns=(1, 2_000_000_000))
    assert server.static_url("a.js") != first


def test_library_module_does_not_depend_on_the_web_layer():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(libmod))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    assert not any(m and m.split(".")[0] in ("fastapi", "starlette", "server") for m in imported)
