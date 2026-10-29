import re
import shutil
from pathlib import Path

import pytest
from conftest import build_epub, load_json
from ebooklib import epub

from converter import cli, convert, images, storage, toc


def test_spine_order_and_toc_splits(converted):
    book, _ = converted
    # spine order (the file the TOC omits comes before ch2), with ch1 split at its anchors
    assert [c.title for c in book.spine][1:] == ["Part A", "Part B", "Not in TOC", "Two"]
    assert [c.order for c in book.spine] == list(range(len(book.spine)))


def test_anchor_split_separates_content(converted):
    book, _ = converted
    part_a = next(c for c in book.spine if c.title == "Part A")
    part_b = next(c for c in book.spine if c.title == "Part B")
    assert "alpha text" in part_a.text and "beta text" not in part_a.text
    assert "beta text" in part_b.text and "alpha text" not in part_b.text


def test_toc_entries_split_href_and_anchor(converted):
    book, _ = converted
    entry = next(e for e in book.toc if e.title == "Part B")
    assert entry.file_href == "text/ch1.xhtml"
    assert entry.anchor == "b"


def test_internal_links_become_chapter_placeholders(converted):
    book, _ = converted
    two = next(c for c in book.spine if c.title == "Two")
    part_b_index = next(i for i, c in enumerate(book.spine) if c.title == "Part B")
    assert f"__CHAPTER__{part_b_index}#b" in two.content
    assert "__INTERNAL_LINK__" not in two.content
    assert 'href="#local"' in two.content


def test_hostile_markup_is_neutralised(converted):
    book, _ = converted
    two = next(c for c in book.spine if c.title == "Two").content.lower()
    for bad in ("<script", "onclick", "javascript:", "vbscript:", "data:text/html", "<iframe", "secret comment"):
        assert bad not in two
    assert "java%73cript" not in two
    assert "https://example.com/ok" in two


def test_colliding_image_names_stay_distinct(converted):
    book, out = converted
    part_a = next(c for c in book.spine if c.title == "Part A").content
    srcs = re.findall(r'<img[^>]*src="([^"]*)"', part_a)
    assert len(srcs) == len(set(srcs)) == 4
    contents = [(out / s).read_bytes()[8:].decode() for s in srcs]
    assert contents == ["A1", "B1", "UP", "LOW"]


def test_svg_cover_wrapper_points_at_extracted_image(converted):
    book, out = converted
    cover_page = book.spine[0].content
    href = re.search(r'xlink:href="([^"]*)"', cover_page).group(1)
    assert href.startswith("images/") and (out / href).is_file()
    assert book.metadata.cover_image and (out / book.metadata.cover_image).is_file()


def test_every_image_reference_exists(converted):
    book, out = converted
    for chapter in book.spine:
        for ref in re.findall(r'(?:src|href)="(images/[^"]*)"', chapter.content):
            assert (out / ref).is_file(), ref


def test_unique_image_name_collisions():
    used: set = set()
    names = [images._unique_image_name(n, used) for n in ("a/1.png", "b/1.png", "x y.png", "xy.png", "P.png", "p.png")]
    assert names == ["1.png", "1_2.png", "xy.png", "xy_2.png", "P.png", "p_2.png"]


def test_metadata(converted):
    book, _ = converted
    assert book.metadata.title == "Test Book"
    assert book.metadata.authors == ["Test Author"]
    assert book.source_file == "test.epub"


def test_failed_conversion_keeps_previous_output(tmp_path, epub_path):
    out = tmp_path / "Book_data"
    convert.convert_epub(str(epub_path), str(out))
    before = (out / "book.json").read_bytes()

    broken = tmp_path / "broken.epub"
    broken.write_text("not a zip")
    with pytest.raises(epub.EpubException):
        convert.convert_epub(str(broken), str(out))

    assert (out / "book.json").read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Book_data", "broken.epub"]  # no staging/backup left


def test_reconversion_replaces_output(tmp_path, epub_path):
    out = tmp_path / "Book_data"
    convert.convert_epub(str(epub_path), str(out))
    (out / "stale.txt").write_text("old")
    convert.convert_epub(str(epub_path), str(out))
    assert not (out / "stale.txt").exists()
    assert load_json(out)["metadata"]["title"] == "Test Book"


def test_fallback_toc_follows_spine_order(tmp_path):
    book = epub.read_epub(str(build_epub(tmp_path / "t.epub")))
    hrefs = [e.href for e in toc.get_fallback_toc(book)]
    assert hrefs == ["cover.xhtml", "text/ch1.xhtml", "text/extra.xhtml", "text/ch2.xhtml"]


def test_cli_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["reader3.py"]) == 1
    assert cli.main(["reader3.py", "missing.epub"]) == 1
    (tmp_path / "bad.epub").write_text("nope")
    assert cli.main(["reader3.py", "bad.epub"]) == 1
    assert "Error" in capsys.readouterr().err
    assert not Path("library/bad_data").exists()


@pytest.fixture
def lib(tmp_path, epub_path):
    d = tmp_path / "lib"
    d.mkdir()
    shutil.copy(epub_path, d / "Book.epub")
    assert cli.main(["reader3.py", str(d / "Book.epub")], library_dir=str(d)) == 0
    return d


def test_cli_list(lib, capsys):
    capsys.readouterr()
    assert cli.main(["reader3.py", "list"], library_dir=str(lib)) == 0
    out = capsys.readouterr().out
    assert "Book" in out and "Test Book" in out and "[epub]" in out


def test_cli_list_empty(tmp_path, capsys):
    assert cli.main(["reader3.py", "list"], library_dir=str(tmp_path / "none")) == 0
    assert "No books" in capsys.readouterr().out


def test_cli_reprocess(lib):
    stale = lib / "Book_data" / "stale.txt"
    stale.write_text("old")
    assert cli.main(["reader3.py", "reprocess", "Book"], library_dir=str(lib)) == 0
    assert not stale.exists()
    assert cli.main(["reader3.py", "reprocess", "--all"], library_dir=str(lib)) == 0


def test_cli_reprocess_without_epub_keeps_data(lib, capsys):
    (lib / "Book.epub").unlink()
    assert cli.main(["reader3.py", "reprocess", "Book"], library_dir=str(lib)) == 1
    assert "not found" in capsys.readouterr().err
    assert (lib / "Book_data" / "book.json").exists()


def test_cli_remove(lib):
    assert cli.main(["reader3.py", "remove", "Book"], library_dir=str(lib)) == 0
    assert not (lib / "Book_data").exists()
    assert (lib / "Book.epub").exists()
    assert cli.main(["reader3.py", "remove", "Book"], library_dir=str(lib)) == 1


@pytest.mark.parametrize("name", ["..", "../x", "a/b", ""])
def test_cli_remove_rejects_unsafe_names(lib, name):
    assert cli.main(["reader3.py", "remove", name], library_dir=str(lib)) == 1
    assert (lib / "Book_data").exists()


def test_cli_list_and_reprocess_legacy_book(lib, capsys):
    (lib / "Book_data" / "book.json").unlink()
    (lib / "Book_data" / "meta.json").unlink()
    (lib / "Book_data" / "book.pkl").write_bytes(b"old")
    capsys.readouterr()
    assert cli.main(["reader3.py", "list"], library_dir=str(lib)) == 0
    assert "run reprocess" in capsys.readouterr().out
    assert cli.main(["reader3.py", "reprocess", "--all"], library_dir=str(lib)) == 0
    assert not (lib / "Book_data" / "book.pkl").exists()
    assert storage.read_summary(str(lib / "Book_data"))["title"] == "Test Book"


def test_meta_summary(converted):
    _, out = converted
    meta = storage.read_summary(str(out))
    assert meta["title"] == "Test Book" and meta["authors"] == ["Test Author"]
    assert meta["chapters"] > 0
