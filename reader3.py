"""
Converts EPUB files into a library of books that server.py can serve.

    uv run reader3.py book.epub        see `uv run reader3.py` for all commands
"""

import sys

from converter.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv))
