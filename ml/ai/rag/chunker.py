"""
Splitting documents into retrievable chunks.

Retrieval quality is decided here more than anywhere else in the pipeline. The
chunker splits on sentence boundaries rather than a fixed character count, so
a chunk never begins mid-clause, and overlaps consecutive chunks so a fact that
straddles a boundary is still fully present in at least one chunk.

Page numbers survive chunking so every retrieved passage can be cited as
"<document>, p. <n>".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ai.common import settings

# Sentence terminator followed by whitespace and a capital/opening character.
# The lookbehind avoids splitting common abbreviations ("Fig.", "e.g.", "vs.").
_SENTENCE_SPLIT = re.compile(
    r"(?<![A-Z][a-z]\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)(?<!\bFig\.)"
    r"(?<!\bDr\.)(?<!\bNo\.)(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])"
)


@dataclass
class Chunk:
    text: str
    index: int
    page: Optional[int] = None
    char_start: int = 0
    metadata: dict = field(default_factory=dict)


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences, keeping terminators attached."""
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    page: Optional[int] = None,
    start_index: int = 0,
) -> list[Chunk]:
    """
    Split text into overlapping, sentence-aligned chunks.

    >>> chunks = chunk_text("One. Two. Three.", chunk_size=10, overlap=0)
    >>> all(c.text for c in chunks)
    True
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap
    # An overlap at or above the chunk size would never advance the cursor.
    overlap = max(0, min(overlap, chunk_size // 2))

    text = text.strip()
    if not text:
        return []

    sentences = split_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    char_cursor = 0
    index = start_index

    for sentence in sentences:
        # A single sentence longer than the budget is hard-split rather than
        # emitted as an oversized chunk that would dominate every embedding.
        if len(sentence) > chunk_size:
            if current:
                chunks.append(
                    Chunk(" ".join(current), index, page, char_cursor)
                )
                index += 1
                char_cursor += current_len
                current, current_len = [], 0
            for i in range(0, len(sentence), chunk_size):
                piece = sentence[i : i + chunk_size]
                chunks.append(Chunk(piece, index, page, char_cursor + i))
                index += 1
            char_cursor += len(sentence)
            continue

        if current_len + len(sentence) + 1 > chunk_size and current:
            chunks.append(Chunk(" ".join(current), index, page, char_cursor))
            index += 1

            # Carry the tail of this chunk into the next one.
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            char_cursor += max(current_len - tail_len, 1)
            current = tail
            current_len = tail_len

        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(Chunk(" ".join(current), index, page, char_cursor))

    return [c for c in chunks if c.text.strip()]


def chunk_document(pages, chunk_size: Optional[int] = None, overlap: Optional[int] = None):
    """Chunk a :class:`ParsedDocument`'s pages, preserving page numbers."""
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(
            chunk_text(
                page.text,
                chunk_size=chunk_size,
                overlap=overlap,
                page=page.number,
                start_index=len(chunks),
            )
        )
    return chunks
