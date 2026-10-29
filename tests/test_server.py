import json
import os
import re

import pytest

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


def test_chapter_links_are_rewritten(client, converted):
    book, _ = converted
    two_index = next(i for i, c in enumerate(book.spine) if c.title == "Two")
    html = client.get(f"/read/Test_data/{two_index}").text
    assert "__CHAPTER__" not in html
    assert re.search(r'href="/read/Test_data/\d+#b"', html)


def test_images_and_cover_served(client, converted):
    book, _ = converted
    assert client.get("/cover/Test_data").status_code == 200
    image = book.metadata.cover_image.split("/", 1)[1]
    r = client.get(f"/read/Test_data/images/{image}")
    assert r.status_code == 200 and r.content.startswith(b"\x89PNG")
    assert client.get("/read/Test_data/images/missing.png").status_code == 404
    assert client.get("/read/Test_data/images/..%2Fbook.json").status_code == 404


@pytest.mark.parametrize("path", [
    "/read/..%2F..%2Fetc/0", "/read/%2e%2e/0", "/read/library/0", "/read/Nope_data/0",
    "/cover/%2e%2e", "/cover/Nope_data", "/read/Nope_data", "/read/Test%00_data/0",
])
def test_invalid_book_ids_are_404(client, path):
    assert client.get(path, follow_redirects=False).status_code == 404


def test_folder_without_data_suffix_is_not_served(client, library):
    (library / "plain").mkdir()
    for name in ("book.json", "meta.json"):
        (library / "plain" / name).write_bytes((library / "Test_data" / name).read_bytes())
    assert client.get("/read/plain/0").status_code == 404
    assert client.get("/cover/plain").status_code == 404


def test_is_valid_book_id():
    assert server._is_valid_book_id("Book_data")
    assert server._is_valid_book_id("My Book #1_data")
    for bad in ("", "Book", "../x_data", "a/b_data", "a\\b_data", "x\x00_data", ".."):
        assert not server._is_valid_book_id(bad)


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
    monkeypatch.setattr(server, "load_book_file", boom)
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
    for path, marker in [("theme.js", "toggleTheme"), ("reader.js", "reader-config"),
                         ("reader.css", "--font-scale"), ("library.js", "read-btn"), ("library.css", "book-grid")]:
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
    assert config == {"bookId": "Test_data", "chapter": 2,
                      "prevUrl": "/read/Test_data/1", "nextUrl": "/read/Test_data/3"}
    assert '/static/reader.js' in html and '/static/theme.js' in html


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
    first = server.load_book_cached("Test_data")
    assert server.load_book_cached("Test_data") is first

    path = library / "Test_data" / "book.json"
    data = json.loads(path.read_text())
    data["metadata"]["title"] = "Changed Title"
    path.write_text(json.dumps(data))
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    assert server.load_book_cached("Test_data").metadata.title == "Changed Title"
