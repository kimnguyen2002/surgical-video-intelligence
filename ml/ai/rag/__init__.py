"""Local retrieval-augmented generation over a user-supplied document corpus."""

from .documents import ParsedDocument, extract_text, supported_extensions
from .chunker import Chunk, chunk_text
from .index import RagIndex, rag_index

__all__ = [
    "ParsedDocument",
    "extract_text",
    "supported_extensions",
    "Chunk",
    "chunk_text",
    "RagIndex",
    "rag_index",
]
