"""Provider interface shared by every language model backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Iterable, Optional


@dataclass
class ChatTurn:
    role: str  # system | user | assistant
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    #: True when the answer came from a real language model rather than the
    #: deterministic knowledge engine. The UI shows this so a user is never
    #: misled about what produced an answer.
    generative: bool = True
    latency_ms: float = 0.0
    usage: dict = field(default_factory=dict)
    error: Optional[str] = None


class LLMProvider:
    """Base class. Subclasses implement :meth:`generate` and optionally :meth:`stream`."""

    name = "base"
    model = ""
    generative = True

    def is_available(self) -> bool:
        raise NotImplementedError

    async def generate(self, messages: list[ChatTurn], **kwargs) -> LLMResponse:
        raise NotImplementedError

    async def stream(self, messages: list[ChatTurn], **kwargs) -> AsyncIterator[str]:
        """
        Default streaming: generate the whole answer, then emit it in
        sentence-sized pieces. Providers with native streaming override this.
        """
        response = await self.generate(messages, **kwargs)
        for piece in _sentence_pieces(response.text):
            yield piece

    def info(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "available": self.is_available(),
            "generative": self.generative,
        }


def _sentence_pieces(text: str, size: int = 90) -> Iterable[str]:
    """Chunk text at word boundaries so streamed output never splits a word."""
    words = text.split(" ")
    buffer = ""
    for word in words:
        if len(buffer) + len(word) + 1 > size and buffer:
            yield buffer + " "
            buffer = word
        else:
            buffer = f"{buffer} {word}".strip()
    if buffer:
        yield buffer
