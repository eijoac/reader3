"""Image extraction: unique file names, cover detection, and src rewriting."""

import logging
import os
import posixpath
from typing import Dict, List, Optional
from urllib.parse import unquote

import ebooklib
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


def _unique_image_name(item_name: str, used_names: set, prefix: str = "") -> str:
    """
    Filesystem-safe name for an image, unique within the book.
    Images in different EPUB folders can share a basename (a/1.png, b/1.png), and
    different names can sanitize to the same string, so a numeric suffix is added on collision.
    """
    clean = "".join(c for c in os.path.basename(item_name) if c.isalnum() or c in '._-').strip() or "image"
    stem, ext = os.path.splitext(clean)
    name = prefix + clean
    counter = 2
    while name.lower() in used_names:  # case-insensitive: the output may live on a case-insensitive filesystem
        name = f"{prefix}{stem}_{counter}{ext}"
        counter += 1
    used_names.add(name.lower())
    return name


def _save_cover(item, images_dir: str, used_names: set) -> str:
    name = _unique_image_name(item.get_name(), used_names, prefix="cover_")
    with open(os.path.join(images_dir, name), 'wb') as f:
        f.write(item.get_content())
    return f"images/{name}"


def extract_cover_image(book_obj, images_dir: str, used_names: set) -> Optional[str]:
    """
    Extracts the cover image from the epub.
    Tries multiple methods to find the cover.
    Returns the relative path to the saved cover image.
    """
    # Method 1: Try to get cover from metadata
    try:
        cover_item = None
        for item in book_obj.get_items():
            if item.get_type() == ebooklib.ITEM_COVER:
                cover_item = item
                break
        
        if cover_item:
            return _save_cover(cover_item, images_dir, used_names)
    except Exception as e:
        log.warning("Cover lookup via ITEM_COVER failed: %s", e)
    
    # Method 2: Check metadata for cover reference
    try:
        cover_meta = book_obj.get_metadata('OPF', 'cover')
        if cover_meta:
            cover_id = cover_meta[0][1].get('content') if cover_meta[0][1] else cover_meta[0][0]
            if cover_id:
                for item in book_obj.get_items():
                    if item.get_id() == cover_id or item.get_name().endswith(cover_id):
                        return _save_cover(item, images_dir, used_names)
    except Exception as e:
        log.warning("Cover lookup via metadata failed: %s", e)
    
    # Method 3: Look for common cover image names
    cover_patterns = ['cover', 'Cover', 'COVER', 'front']
    for item in book_obj.get_items():
        if item.get_type() == ebooklib.ITEM_IMAGE:
            name = item.get_name().lower()
            if any(pattern.lower() in name for pattern in cover_patterns):
                return _save_cover(item, images_dir, used_names)
    
    # Method 4: Get the first image as a fallback
    for item in book_obj.get_items():
        if item.get_type() == ebooklib.ITEM_IMAGE:
            log.warning("Using first image as cover (no explicit cover found)")
            return _save_cover(item, images_dir, used_names)
    
    return None


def extract_images(book_obj, images_dir: str, used_names: set) -> Dict[str, str]:
    """
    Writes every image to `images_dir` and returns {epub path -> "images/<file>"}.
    A bare basename also maps to its image, but only when that basename is unambiguous.
    """
    image_map: Dict[str, str] = {}
    basename_targets: Dict[str, List[str]] = {}
    for item in book_obj.get_items():
        if item.get_type() in (ebooklib.ITEM_IMAGE, ebooklib.ITEM_COVER):
            safe_fname = _unique_image_name(item.get_name(), used_names)
            with open(os.path.join(images_dir, safe_fname), 'wb') as f:
                f.write(item.get_content())
            rel_path = f"images/{safe_fname}"
            image_map[item.get_name()] = rel_path
            basename_targets.setdefault(os.path.basename(item.get_name()), []).append(rel_path)

    # Basename-only lookup is a fallback for sloppy src attributes
    for basename, targets in basename_targets.items():
        if len(targets) == 1:
            image_map.setdefault(basename, targets[0])
    return image_map


def rewrite_image_sources(soup: BeautifulSoup, doc_name: str, image_map: Dict[str, str]) -> None:
    """Points <img src> and SVG <image xlink:href|href> (cover-page wrappers) at the extracted files."""
    for img in soup.find_all(['img', 'image']):
        attrs = ('src',) if img.name == 'img' else ('xlink:href', 'href')
        for attr in attrs:
            src = img.get(attr, '')
            if not src:
                continue
            src_decoded = unquote(src).split('#', 1)[0].split('?', 1)[0]
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(doc_name), src_decoded))
            for candidate in (resolved, src_decoded, posixpath.basename(src_decoded)):
                if candidate in image_map:
                    img[attr] = image_map[candidate]
                    break
