"""Turn a document file into numbered lines, so every quote can point to a line."""
from pathlib import Path

from pypdf import PdfReader


class NoTextLayer(Exception):
    """The PDF has no extractable text: a scanned document."""


def read_lines(path: Path) -> tuple[list[str], int]:
    """Return (lines, pages). Text files are read as UTF-8; PDFs through pypdf, pages joined in order."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        lines: list[str] = []
        for page in reader.pages:
            lines.extend((page.extract_text() or "").splitlines())
        if not any(line.strip() for line in lines):
            raise NoTextLayer("no text layer: scanned document, OCR not supported")
        return [line.rstrip() for line in lines], len(reader.pages)
    return path.read_text(encoding="utf-8").splitlines(), 1
