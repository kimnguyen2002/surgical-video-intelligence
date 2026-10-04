"""Pluggable vector index used by both the RAG corpus and temporal memory."""

from .store import SearchResult, VectorStore

__all__ = ["VectorStore", "SearchResult"]
