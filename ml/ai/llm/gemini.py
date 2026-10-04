"""
Gemini provider (optional).

The platform is open-source and local-first: Ollama is the default and nothing
requires a cloud key. This provider exists because the project was scaffolded
in Google AI Studio and a ``GEMINI_API_KEY`` is already injected there, so it
works out of the box in that environment. It activates *only* when the key is
present, and the UI always labels which provider produced an answer.
"""

from __future__ import annotations

import logging
import time
from typing import AsyncIterator, Optional

from ai.common import optional_import, settings

from .base import ChatTurn, LLMProvider, LLMResponse

logger = logging.getLogger("charlie.llm.gemini")


class GeminiProvider(LLMProvider):
    name = "gemini"
    generative = True

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model = model or settings.gemini_model
        self._client = None

    def is_available(self) -> bool:
        # AI Studio injects the literal placeholder when no secret is set.
        if not self.api_key or self.api_key.startswith("MY_"):
            return False
        return optional_import("google_genai") is not None

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _split(messages: list[ChatTurn]):
        """Gemini takes the system prompt separately from the turn history."""
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = [
            {
                "role": "model" if m.role == "assistant" else "user",
                "parts": [{"text": m.content}],
            }
            for m in messages
            if m.role != "system"
        ]
        return system, contents

    def _config(self, system: str, **kwargs):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=kwargs.get("temperature", settings.llm_temperature),
            max_output_tokens=kwargs.get("max_tokens", settings.llm_max_tokens),
        )

    async def generate(self, messages: list[ChatTurn], **kwargs) -> LLMResponse:
        started = time.time()
        model = kwargs.get("model", self.model)
        try:
            client = self._get_client()
            system, contents = self._split(messages)
            response = await client.aio.models.generate_content(
                model=model, contents=contents, config=self._config(system, **kwargs)
            )
            usage = {}
            meta = getattr(response, "usage_metadata", None)
            if meta is not None:
                usage = {
                    "prompt_tokens": getattr(meta, "prompt_token_count", 0),
                    "completion_tokens": getattr(meta, "candidates_token_count", 0),
                }
            return LLMResponse(
                text=(response.text or "").strip(),
                provider=self.name,
                model=model,
                latency_ms=round((time.time() - started) * 1000, 1),
                usage=usage,
            )
        except Exception as exc:
            logger.error("Gemini generation failed: %s", exc)
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=round((time.time() - started) * 1000, 1),
                error=str(exc),
            )

    async def stream(self, messages: list[ChatTurn], **kwargs) -> AsyncIterator[str]:
        model = kwargs.get("model", self.model)
        try:
            client = self._get_client()
            system, contents = self._split(messages)
            stream = await client.aio.models.generate_content_stream(
                model=model, contents=contents, config=self._config(system, **kwargs)
            )
            async for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except Exception as exc:
            logger.error("Gemini streaming failed: %s", exc)
            yield f"\n\n_(Gemini stream interrupted: {exc})_"
