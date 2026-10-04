"""Text and vision embedding backends with graceful degradation."""

from .text import TextEmbedder, get_text_embedder
from .vision import FrameEmbedder, get_frame_embedder

__all__ = [
    "TextEmbedder",
    "get_text_embedder",
    "FrameEmbedder",
    "get_frame_embedder",
]
