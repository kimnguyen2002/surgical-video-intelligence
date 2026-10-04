"""
The RAG corpus: ingest documents, retrieve grounded passages.

Retrieval is deliberately conservative. Passages below a relevance floor are
dropped rather than padded into the context, and :meth:`RagIndex.retrieve`
returns the verbatim excerpt alongside its document and page. The generation
layer is then required to present those excerpts as *retrieved evidence*,
separate from its own explanation — the platform must never manufacture a
citation, so a citation only ever names a passage that was actually indexed
and actually scored above the floor.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from ai.common import settings
from ai.embeddings import get_text_embedder
from ai.vectorstore import VectorStore

from .chunker import chunk_document
from .documents import ParsedDocument, extract_text

logger = logging.getLogger("charlie.rag.index")


@dataclass
class DocumentRecord:
    doc_id: str
    filename: str
    format: str
    chunk_count: int
    char_count: int
    uploaded_at: float = field(default_factory=time.time)
    specialty: str = ""
    tags: list[str] = field(default_factory=list)
    pages: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievedPassage:
    document: str
    doc_id: str
    chunk_id: str
    score: float
    excerpt: str
    page: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def citation(self) -> str:
        return f"{self.document}, p. {self.page}" if self.page else self.document


class RagIndex:
    """A persistent, searchable corpus of user-supplied documents."""

    def __init__(self, directory: Optional[Path] = None):
        self.directory = Path(directory or settings.documents_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.directory / "manifest.json"
        self._lock = threading.RLock()
        self._documents: dict[str, DocumentRecord] = {}

        embedder = get_text_embedder()
        self._store = VectorStore("rag_chunks", dim=embedder.dim)
        self._load_manifest()

    # -- ingestion ---------------------------------------------------------
    def add_document(
        self,
        filename: str,
        data: bytes,
        specialty: str = "",
        tags: Optional[list[str]] = None,
    ) -> dict:
        """
        Parse, chunk, embed, and index one uploaded document.

        Returns a result dict with ``ok`` plus either the document record or a
        human-readable ``error``.
        """
        parsed: ParsedDocument = extract_text(filename, data)
        if parsed.error:
            return {"ok": False, "error": parsed.error, "filename": filename}
        if not parsed.pages:
            return {
                "ok": False,
                "error": "No text could be extracted from this file.",
                "filename": filename,
            }

        # Content-addressed id: re-uploading the same file replaces its chunks
        # instead of duplicating them across the corpus.
        doc_id = hashlib.sha256(data).hexdigest()[:16]
        chunks = chunk_document(parsed.pages)
        if not chunks:
            return {"ok": False, "error": "Document produced no chunks.", "filename": filename}

        embedder = get_text_embedder()
        vectors = embedder.encode([c.text for c in chunks])

        batch = []
        for chunk, vector in zip(chunks, vectors):
            chunk_id = f"{doc_id}:{chunk.index:05d}"
            batch.append(
                (
                    chunk_id,
                    vector,
                    {
                        "doc_id": doc_id,
                        "filename": parsed.filename,
                        "page": chunk.page,
                        "chunk_index": chunk.index,
                        "specialty": specialty,
                        "tags": tags or [],
                    },
                    chunk.text,
                )
            )

        with self._lock:
            if doc_id in self._documents:
                self._remove_chunks(doc_id)
            self._store.add_many(batch)
            record = DocumentRecord(
                doc_id=doc_id,
                filename=parsed.filename,
                format=parsed.format,
                chunk_count=len(chunks),
                char_count=parsed.char_count,
                specialty=specialty,
                tags=tags or [],
                pages=len(parsed.pages),
            )
            self._documents[doc_id] = record
            self._save_manifest()

        logger.info(
            "Indexed '%s' → %d chunks (%d chars)",
            parsed.filename,
            len(chunks),
            parsed.char_count,
        )
        return {"ok": True, "document": record.to_dict()}

    def remove_document(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self._documents:
                return False
            self._remove_chunks(doc_id)
            del self._documents[doc_id]
            self._save_manifest()
        return True

    def _remove_chunks(self, doc_id: str) -> None:
        ids = [
            m["id"] for m in self._store.all_metadata() if m.get("doc_id") == doc_id
        ]
        if ids:
            self._store.delete(ids)

    # -- retrieval ---------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        specialty: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> list[RetrievedPassage]:
        """Retrieve the passages most relevant to a query."""
        if not self._documents:
            return []

        top_k = top_k or settings.rag_top_k
        floor = settings.rag_min_score if min_score is None else min_score

        vector = get_text_embedder().encode_one(query)
        where = {"specialty": specialty} if specialty else None
        results = self._store.search(vector, top_k=top_k, where=where, min_score=floor)

        # If a specialty filter produced nothing, fall back to the whole corpus
        # rather than answering with no evidence at all.
        if not results and where:
            results = self._store.search(vector, top_k=top_k, min_score=floor)

        passages = []
        for result in results:
            meta = result.metadata
            passages.append(
                RetrievedPassage(
                    document=meta.get("filename", "unknown"),
                    doc_id=meta.get("doc_id", ""),
                    chunk_id=result.id,
                    score=result.score,
                    excerpt=result.text[:1200],
                    page=meta.get("page"),
                )
            )
        return passages

    def build_context(self, passages: list[RetrievedPassage], budget: int = 4000) -> str:
        """Format retrieved passages for injection into an LLM prompt."""
        blocks, used = [], 0
        for i, passage in enumerate(passages, start=1):
            block = f"[{i}] {passage.citation()}\n{passage.excerpt}"
            if used + len(block) > budget:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    # -- introspection -----------------------------------------------------
    def list_documents(self) -> list[dict]:
        with self._lock:
            return sorted(
                (d.to_dict() for d in self._documents.values()),
                key=lambda d: d["uploaded_at"],
                reverse=True,
            )

    def stats(self) -> dict:
        embedder = get_text_embedder()
        return {
            "documents": len(self._documents),
            "chunks": len(self._store),
            "embedder": embedder.info(),
            "store": self._store.stats(),
        }

    # -- persistence -------------------------------------------------------
    def _save_manifest(self) -> None:
        try:
            self._manifest_path.write_text(
                json.dumps(
                    {k: v.to_dict() for k, v in self._documents.items()},
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Could not write RAG manifest: %s", exc)

    def _load_manifest(self) -> None:
        if not self._manifest_path.exists():
            return
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Unreadable RAG manifest (%s); starting empty.", exc)
            return
        for doc_id, record in data.items():
            try:
                self._documents[doc_id] = DocumentRecord(**record)
            except TypeError:
                continue
        logger.info("Restored RAG corpus: %d documents", len(self._documents))


#: Process-wide corpus used by the API layer.
rag_index = RagIndex()
