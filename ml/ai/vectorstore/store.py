"""
Persistent vector index with three interchangeable backends.

``faiss``   IndexFlatIP over L2-normalised vectors — exact cosine, SIMD-fast,
            the right choice past ~100k vectors.
``numpy``   A single matrix multiply. Exact, and comfortably sub-millisecond
            at the scale a single surgical procedure produces.
``python``  Pure standard library. Slower, but it means the platform has no
            hard dependency at all.

All three implement identical semantics, so the backend is an optimisation
detail rather than a behavioural one. Vectors are always stored L2-normalised,
which makes inner product equal cosine similarity and puts every score on a
comparable 0–1 scale.

Persistence is a JSONL sidecar of records plus a backend-specific vector file,
so an index can be inspected, diffed, and version-controlled by researchers.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ai.common import optional_import, settings

logger = logging.getLogger("charlie.vectorstore")


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict = field(default_factory=dict)
    text: str = ""


class VectorStore:
    """
    A named, persistent collection of vectors with attached metadata.

    >>> store = VectorStore("demo", dim=4)
    >>> store.add("a", [1, 0, 0, 0], {"kind": "x"}, text="hello")
    >>> store.search([1, 0, 0, 0], top_k=1)[0].id
    'a'
    """

    def __init__(
        self,
        name: str,
        dim: int,
        directory: Optional[Path] = None,
        backend: Optional[str] = None,
    ):
        self.name = name
        self.dim = dim
        self.directory = Path(directory or settings.vector_dir) / name
        self.directory.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._metadata: dict[str, dict] = {}
        self._texts: dict[str, str] = {}
        self._id_to_row: dict[str, int] = {}

        self.backend = self._select_backend(backend or settings.vector_backend)
        self._np = optional_import("numpy") if self.backend in {"numpy", "faiss"} else None
        self._faiss_index = None
        self._matrix = None  # cached numpy matrix, invalidated on write

        self._load()

    # -- backend selection ------------------------------------------------
    @staticmethod
    def _select_backend(requested: str) -> str:
        if requested == "faiss":
            return "faiss" if optional_import("faiss") else "numpy"
        if requested == "numpy":
            return "numpy" if optional_import("numpy") else "python"
        if requested == "python":
            return "python"
        # auto
        if optional_import("faiss") is not None and optional_import("numpy") is not None:
            return "faiss"
        if optional_import("numpy") is not None:
            return "numpy"
        return "python"

    # -- paths -------------------------------------------------------------
    @property
    def _records_path(self) -> Path:
        return self.directory / "records.jsonl"

    @property
    def _vectors_path(self) -> Path:
        return self.directory / "vectors.jsonl"

    # -- mutation ----------------------------------------------------------
    def add(
        self,
        id: str,
        vector: Iterable[float],
        metadata: Optional[dict] = None,
        text: str = "",
        persist: bool = True,
    ) -> None:
        self.add_many([(id, vector, metadata or {}, text)], persist=persist)

    def add_many(
        self,
        items: Iterable[tuple[str, Iterable[float], dict, str]],
        persist: bool = True,
    ) -> int:
        """Insert (or replace) a batch of vectors. Returns the number written."""
        written = 0
        with self._lock:
            for id_, vector, metadata, text in items:
                vec = _normalise(list(map(float, vector)), self.dim)
                if id_ in self._id_to_row:
                    row = self._id_to_row[id_]
                    self._vectors[row] = vec
                else:
                    self._id_to_row[id_] = len(self._ids)
                    self._ids.append(id_)
                    self._vectors.append(vec)
                self._metadata[id_] = metadata or {}
                self._texts[id_] = text
                written += 1
            self._invalidate()
            if persist:
                self._save()
        return written

    def delete(self, ids: Iterable[str]) -> int:
        """Remove records by id and compact the index."""
        target = set(ids)
        with self._lock:
            keep = [i for i, id_ in enumerate(self._ids) if id_ not in target]
            removed = len(self._ids) - len(keep)
            if not removed:
                return 0
            self._vectors = [self._vectors[i] for i in keep]
            self._ids = [self._ids[i] for i in keep]
            for id_ in target:
                self._metadata.pop(id_, None)
                self._texts.pop(id_, None)
            self._id_to_row = {id_: i for i, id_ in enumerate(self._ids)}
            self._invalidate()
            self._save()
        return removed

    def clear(self) -> None:
        with self._lock:
            self._ids.clear()
            self._vectors.clear()
            self._metadata.clear()
            self._texts.clear()
            self._id_to_row.clear()
            self._invalidate()
            self._save()

    # -- query -------------------------------------------------------------
    def search(
        self,
        query: Iterable[float],
        top_k: int = 5,
        where: Optional[dict] = None,
        min_score: float = -1.0,
    ) -> list[SearchResult]:
        """
        Return the ``top_k`` most similar records.

        ``where`` filters on exact metadata equality; a list value matches if
        the record's value is in the list. Filtering is applied *before*
        ranking so top_k always returns k matching records when they exist.
        """
        with self._lock:
            if not self._ids:
                return []

            q = _normalise(list(map(float, query)), self.dim)
            candidates = self._filter_rows(where)
            if not candidates:
                return []

            scores = self._score(q, candidates)
            ranked = sorted(scores, key=lambda p: p[1], reverse=True)[:top_k]

            results = []
            for row, score in ranked:
                if score < min_score:
                    continue
                id_ = self._ids[row]
                results.append(
                    SearchResult(
                        id=id_,
                        score=round(float(score), 6),
                        metadata=self._metadata.get(id_, {}),
                        text=self._texts.get(id_, ""),
                    )
                )
            return results

    def _filter_rows(self, where: Optional[dict]) -> list[int]:
        if not where:
            return list(range(len(self._ids)))
        rows = []
        for row, id_ in enumerate(self._ids):
            meta = self._metadata.get(id_, {})
            ok = True
            for key, expected in where.items():
                actual = meta.get(key)
                if isinstance(expected, (list, tuple, set)):
                    if actual not in expected:
                        ok = False
                        break
                elif actual != expected:
                    ok = False
                    break
            if ok:
                rows.append(row)
        return rows

    def _score(self, query: list[float], rows: list[int]) -> list[tuple[int, float]]:
        # FAISS only helps when scanning everything; a filtered subset is
        # cheaper to score directly.
        if self.backend == "faiss" and len(rows) == len(self._ids):
            faiss_scores = self._faiss_search(query, len(rows))
            if faiss_scores is not None:
                return faiss_scores

        if self._np is not None:
            np = self._np
            matrix = self._as_matrix()
            if matrix is not None:
                sub = matrix[rows] if len(rows) != len(self._ids) else matrix
                sims = sub @ np.asarray(query, dtype="float32")
                return list(zip(rows, sims.tolist()))

        return [(row, _dot(query, self._vectors[row])) for row in rows]

    def _faiss_search(self, query: list[float], k: int) -> Optional[list[tuple[int, float]]]:
        faiss = optional_import("faiss")
        np = self._np
        if faiss is None or np is None:
            return None
        try:
            if self._faiss_index is None:
                index = faiss.IndexFlatIP(self.dim)
                index.add(np.asarray(self._vectors, dtype="float32"))
                self._faiss_index = index
            q = np.asarray([query], dtype="float32")
            scores, indices = self._faiss_index.search(q, min(k, len(self._ids)))
            return [
                (int(i), float(s))
                for i, s in zip(indices[0], scores[0])
                if i >= 0
            ]
        except Exception as exc:
            logger.warning("FAISS search failed (%s); falling back to numpy.", exc)
            self._faiss_index = None
            return None

    def _as_matrix(self):
        if self._np is None:
            return None
        if self._matrix is None and self._vectors:
            self._matrix = self._np.asarray(self._vectors, dtype="float32")
        return self._matrix

    def _invalidate(self) -> None:
        self._matrix = None
        self._faiss_index = None

    # -- access ------------------------------------------------------------
    def get(self, id: str) -> Optional[SearchResult]:
        with self._lock:
            if id not in self._id_to_row:
                return None
            return SearchResult(
                id=id,
                score=1.0,
                metadata=self._metadata.get(id, {}),
                text=self._texts.get(id, ""),
            )

    def all_metadata(self) -> list[dict]:
        with self._lock:
            return [{"id": i, **self._metadata.get(i, {})} for i in self._ids]

    def __len__(self) -> int:
        return len(self._ids)

    def stats(self) -> dict:
        return {
            "name": self.name,
            "backend": self.backend,
            "count": len(self._ids),
            "dim": self.dim,
            "path": str(self.directory),
        }

    # -- persistence -------------------------------------------------------
    def _save(self) -> None:
        try:
            with self._records_path.open("w", encoding="utf-8") as fh:
                for id_ in self._ids:
                    fh.write(
                        json.dumps(
                            {
                                "id": id_,
                                "metadata": self._metadata.get(id_, {}),
                                "text": self._texts.get(id_, ""),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            with self._vectors_path.open("w", encoding="utf-8") as fh:
                for id_, vec in zip(self._ids, self._vectors):
                    # Six decimals keeps files readable without measurably
                    # affecting cosine scores.
                    fh.write(json.dumps([round(v, 6) for v in vec]) + "\n")
        except OSError as exc:
            logger.error("Could not persist vector store '%s': %s", self.name, exc)

    def _load(self) -> None:
        if not self._records_path.exists() or not self._vectors_path.exists():
            return
        try:
            with self._records_path.open(encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh if line.strip()]
            with self._vectors_path.open(encoding="utf-8") as fh:
                vectors = [json.loads(line) for line in fh if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Corrupt vector store '%s' (%s); starting empty.", self.name, exc)
            return

        if len(records) != len(vectors):
            logger.error(
                "Vector store '%s' is inconsistent (%d records vs %d vectors); "
                "starting empty.",
                self.name,
                len(records),
                len(vectors),
            )
            return

        for record, vector in zip(records, vectors):
            id_ = record["id"]
            self._id_to_row[id_] = len(self._ids)
            self._ids.append(id_)
            self._vectors.append(_normalise(vector, self.dim))
            self._metadata[id_] = record.get("metadata", {})
            self._texts[id_] = record.get("text", "")

        logger.info("Loaded vector store '%s': %d records", self.name, len(self._ids))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _normalise(vector: list[float], dim: int) -> list[float]:
    """Pad/trim to `dim` and scale to unit length."""
    if len(vector) < dim:
        vector = vector + [0.0] * (dim - len(vector))
    elif len(vector) > dim:
        vector = vector[:dim]
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
