"""
Document parsing for the RAG corpus.

Textbooks, lecture notes, guidelines, papers, slides, and protocols are
converted into page-tagged plain text. Page numbers are preserved wherever the
format carries them so citations can point at a specific page rather than a
whole document — a citation a reader cannot verify is worse than no citation.

Formats without a parser installed are rejected with a message naming the
package that would enable them, rather than silently producing empty text.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ai.common import optional_import

logger = logging.getLogger("charlie.rag.documents")


@dataclass
class Page:
    number: int
    text: str


@dataclass
class ParsedDocument:
    filename: str
    pages: list[Page] = field(default_factory=list)
    format: str = "txt"
    error: Optional[str] = None

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


ALWAYS_SUPPORTED = {".txt", ".md", ".markdown", ".csv", ".json", ".rst", ".html", ".htm"}
OPTIONAL_FORMATS = {
    ".pdf": ("pypdf", "pip install pypdf"),
    ".pptx": ("pptx", "pip install python-pptx"),
    ".docx": ("docx", "pip install python-docx"),
}


def supported_extensions() -> dict:
    """Which upload formats this instance can actually parse."""
    from ai.common import has

    out = {ext: True for ext in sorted(ALWAYS_SUPPORTED)}
    for ext, (cap, install) in OPTIONAL_FORMATS.items():
        out[ext] = has(cap)
    return out


def extract_text(filename: str, data: bytes) -> ParsedDocument:
    """Parse an uploaded file into page-tagged text."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(filename, data)
    if suffix == ".pptx":
        return _parse_pptx(filename, data)
    if suffix == ".docx":
        return _parse_docx(filename, data)
    if suffix in {".html", ".htm"}:
        return _parse_html(filename, data)
    if suffix in ALWAYS_SUPPORTED:
        return _parse_plain(filename, data, suffix)

    return ParsedDocument(
        filename=filename,
        format=suffix or "unknown",
        error=(
            f"Unsupported file type '{suffix or 'unknown'}'. Supported: "
            + ", ".join(sorted(ALWAYS_SUPPORTED | set(OPTIONAL_FORMATS)))
        ),
    )


def _decode(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _parse_plain(filename: str, data: bytes, suffix: str) -> ParsedDocument:
    text = _clean(_decode(data))
    return ParsedDocument(
        filename=filename,
        format=suffix.lstrip("."),
        pages=[Page(number=1, text=text)] if text else [],
    )


def _parse_html(filename: str, data: bytes) -> ParsedDocument:
    raw = _decode(data)
    # Drop script/style bodies before stripping tags, otherwise their contents
    # end up in the corpus as if they were prose.
    raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = (
        raw.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
    )
    return ParsedDocument(
        filename=filename, format="html", pages=[Page(number=1, text=_clean(raw))]
    )


def _parse_pdf(filename: str, data: bytes) -> ParsedDocument:
    pypdf = optional_import("pypdf")
    if pypdf is None:
        return ParsedDocument(
            filename=filename,
            format="pdf",
            error="PDF parsing requires pypdf. Install it with: pip install pypdf",
        )
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = _clean(page.extract_text() or "")
            if text:
                pages.append(Page(number=i, text=text))
        doc = ParsedDocument(filename=filename, format="pdf", pages=pages)
        if not pages:
            doc.error = (
                "No extractable text — this looks like a scanned PDF. "
                "Run OCR on it before uploading."
            )
        return doc
    except Exception as exc:
        logger.error("PDF parse failed for %s: %s", filename, exc)
        return ParsedDocument(filename=filename, format="pdf", error=str(exc))


def _parse_pptx(filename: str, data: bytes) -> ParsedDocument:
    pptx = optional_import("pptx")
    if pptx is None:
        return ParsedDocument(
            filename=filename,
            format="pptx",
            error="PPTX parsing requires python-pptx. Install: pip install python-pptx",
        )
    try:
        presentation = pptx.Presentation(io.BytesIO(data))
        pages = []
        for i, slide in enumerate(presentation.slides, start=1):
            parts = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False):
                    parts.append(shape.text_frame.text)
            text = _clean("\n".join(parts))
            if text:
                pages.append(Page(number=i, text=text))
        return ParsedDocument(filename=filename, format="pptx", pages=pages)
    except Exception as exc:
        logger.error("PPTX parse failed for %s: %s", filename, exc)
        return ParsedDocument(filename=filename, format="pptx", error=str(exc))


def _parse_docx(filename: str, data: bytes) -> ParsedDocument:
    docx = optional_import("docx")
    if docx is None:
        return ParsedDocument(
            filename=filename,
            format="docx",
            error="DOCX parsing requires python-docx. Install: pip install python-docx",
        )
    try:
        document = docx.Document(io.BytesIO(data))
        text = _clean("\n".join(p.text for p in document.paragraphs))
        return ParsedDocument(
            filename=filename,
            format="docx",
            pages=[Page(number=1, text=text)] if text else [],
        )
    except Exception as exc:
        logger.error("DOCX parse failed for %s: %s", filename, exc)
        return ParsedDocument(filename=filename, format="docx", error=str(exc))


def _clean(text: str) -> str:
    """Collapse whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
