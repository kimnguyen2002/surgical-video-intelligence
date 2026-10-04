"""
Text embeddings for the RAG corpus and the temporal memory index.

Two tiers, selected automatically:

``sentence-transformers``
    True dense semantic embeddings (all-MiniLM-L6-v2 by default). Runs fully
    offline once the model is cached.

hashed n-grams (always available)
    A deterministic, dependency-free embedding built from hashed word and
    character n-grams with sub-linear term weighting. This is lexical rather
    than semantic — it will match "cystic artery" to a passage containing
    "cystic artery" but not to one that only says "gallbladder blood supply".
    It is fully offline, needs no model download, and keeps retrieval useful
    on a bare Python install. The active tier is reported by :attr:`backend`
    and surfaced in the UI so retrieval quality is never silently overstated.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from typing import Iterable, Optional, Sequence

from ai.common import optional_import, settings

logger = logging.getLogger("charlie.embeddings.text")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "for", "on", "with", "as", "by", "at", "from", "that", "this", "it", "be",
    "has", "have", "had", "but", "not", "we", "you", "they", "he", "she",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


class TextEmbedder:
    """Encodes text into unit-norm vectors."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.text_embedding_model
        self._model = None
        self._lock = threading.Lock()
        self.backend = "hashed"
        self.dim = settings.hashed_embedding_dim
        self._try_load_dense()

    # -- loading ---------------------------------------------------------
    def _try_load_dense(self) -> None:
        st = optional_import("sentence_transformers")
        if st is None:
            logger.info(
                "sentence-transformers not installed — using offline hashed "
                "n-gram embeddings (lexical retrieval)."
            )
            return
        try:
            self._model = st.SentenceTransformer(self.model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
            self.backend = f"sentence-transformers:{self.model_name}"
            logger.info("Text embedder ready: %s (dim=%d)", self.backend, self.dim)
        except Exception as exc:
            # Most commonly: no network on first run and no cached weights.
            logger.warning(
                "Could not load '%s' (%s) — falling back to hashed embeddings.",
                self.model_name,
                exc,
            )
            self._model = None

    @property
    def is_semantic(self) -> bool:
        """True when embeddings capture meaning rather than surface form."""
        return self._model is not None

    # -- encoding --------------------------------------------------------
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a batch of strings into unit-norm vectors."""
        if not texts:
            return []
        if self._model is not None:
            with self._lock:
                try:
                    vectors = self._model.encode(
                        list(texts),
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    return [list(map(float, v)) for v in vectors]
                except Exception as exc:
                    logger.error("Dense encoding failed (%s); using hashed fallback.", exc)
        return [self._hashed(t) for t in texts]

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]

    # -- hashed fallback -------------------------------------------------
    def _hashed(self, text: str) -> list[float]:
        """
        Deterministic hashed-n-gram embedding.

        Word unigrams and bigrams plus character 4-grams are hashed into a
        fixed-width vector with signed buckets (the hashing trick, which keeps
        collisions unbiased). Counts are damped with 1+log(tf) so a term
        repeated ten times does not swamp the vector, then L2-normalised so
        dot product equals cosine similarity.
        """
        dim = self.dim
        vec = [0.0] * dim
        tokens = _tokenize(text)
        if not tokens:
            return vec

        features: list[str] = list(tokens)
        features += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]

        compact = "".join(tokens)
        features += [compact[i : i + 4] for i in range(0, max(len(compact) - 3, 0), 2)]

        counts: dict[str, int] = {}
        for feature in features:
            counts[feature] = counts.get(feature, 0) + 1

        for feature, count in counts.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def info(self) -> dict:
        return {
            "backend": self.backend,
            "dim": self.dim,
            "semantic": self.is_semantic,
            "note": None
            if self.is_semantic
            else "Lexical retrieval only. `pip install sentence-transformers` for semantic search.",
        }


_INSTANCE: Optional[TextEmbedder] = None
_INSTANCE_LOCK = threading.Lock()


def get_text_embedder() -> TextEmbedder:
    """Process-wide singleton — model loading is expensive."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = TextEmbedder()
    return _INSTANCE


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    """Cosine similarity between two vectors."""
    a = list(a)
    b = list(b)
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
