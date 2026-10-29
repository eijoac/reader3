import os
import pickle
import re
import sys

import pytest

import reader3
import server


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
    assert client.get("/read/Test_data/images/..%2Fbook.pkl").status_code == 404


@pytest.mark.parametrize("path", [
    "/read/..%2F..%2Fetc/0", "/read/%2e%2e/0", "/read/library/0", "/read/Nope_data/0",
    "/cover/%2e%2e", "/cover/Nope_data", "/read/Nope_data", "/read/Test%00_data/0",
])
def test_invalid_book_ids_are_404(client, path):
    assert client.get(path, follow_redirects=False).status_code == 404


def test_folder_without_data_suffix_is_not_served(client, library):
    (library / "plain").mkdir()
    (library / "plain" / "book.pkl").write_bytes((library / "Test_data" / "book.pkl").read_bytes())
    assert client.get("/read/plain/0").status_code == 404
    assert client.get("/cover/plain").status_code == 404


def test_is_valid_book_id():
    assert server._is_valid_book_id("Book_data")
    assert server._is_valid_book_id("My Book #1_data")
    for bad in ("", "Book", "../x_data", "a/b_data", "a\\b_data", "x\x00_data", ".."):
        assert not server._is_valid_book_id(bad)


def test_corrupt_book_is_skipped(client, library):
    (library / "Bad_data").mkdir()
    (library / "Bad_data" / "book.pkl").write_text("garbage")
    r = client.get("/")
    assert r.status_code == 200 and "Test Book" in r.text
    assert client.get("/read/Bad_data/0").status_code == 404


def test_pickle_written_as_main_module_still_loads(library, monkeypatch):
    # `python reader3.py` pickles classes as __main__.X; the server must resolve them from reader3
    classes = [reader3.Book, reader3.BookMetadata, reader3.ChapterContent, reader3.TOCEntry]
    book = pickle.loads((library / "Test_data" / "book.pkl").read_bytes())
    main = sys.modules["__main__"]
    for cls in classes:
        monkeypatch.setattr(cls, "__module__", "__main__")
        monkeypatch.setattr(main, cls.__name__, cls, raising=False)
    data = pickle.dumps(book)
    assert b"__main__" in data

    (library / "Main_data").mkdir()
    (library / "Main_data" / "book.pkl").write_bytes(data)
    assert server.load_book_cached("Main_data").metadata.title == "Test Book"


def test_cache_reloads_when_book_is_reprocessed(library):
    first = server.load_book_cached("Test_data")
    assert server.load_book_cached("Test_data") is first

    pkl = library / "Test_data" / "book.pkl"
    book = pickle.loads(pkl.read_bytes())
    book.metadata.title = "Changed Title"
    pkl.write_bytes(pickle.dumps(book))
    st = pkl.stat()
    os.utime(pkl, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    assert server.load_book_cached("Test_data").metadata.title == "Changed Title"
