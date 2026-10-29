"""The book data model."""

from dataclasses import dataclass, field


@dataclass
class ChapterContent:
    """
    Represents a physical file in the EPUB (Spine Item).
    A single file might contain multiple logical chapters (TOC entries).
    """

    id: str  # Internal ID (e.g., 'item_1')
    href: str  # Filename (e.g., 'part01.html')
    title: str  # Best guess title from file
    content: str  # Cleaned HTML with rewritten image paths
    text: str  # Plain text for search/LLM context
    order: int  # Linear reading order


@dataclass
class TOCEntry:
    """Represents a logical entry in the navigation sidebar."""

    title: str
    href: str  # original href (e.g., 'part01.html#chapter1')
    file_href: str  # just the filename (e.g., 'part01.html')
    anchor: str  # just the anchor (e.g., 'chapter1'), empty if none
    children: list[TOCEntry] = field(default_factory=list)


@dataclass
class BookMetadata:
    """Metadata"""

    title: str
    language: str
    authors: list[str] = field(default_factory=list)
    description: str | None = None
    publisher: str | None = None
    date: str | None = None
    identifiers: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    cover_image: str | None = None  # Relative path to cover image


@dataclass
class Book:
    """The master object, stored as book.json."""

    metadata: BookMetadata
    spine: list[ChapterContent]  # The actual content (linear files)
    toc: list[TOCEntry]  # The navigation tree
    images: dict[str, str]  # Map: original_path -> local_path

    # Meta info
    source_file: str
    processed_at: str
