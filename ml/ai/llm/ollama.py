"""
Ollama provider — the fully-offline default.

Talks to a local Ollama server over plain HTTP, so no Python client library is
required and the platform keeps working when the machine has no internet
access at all. Any Ollama-served open model (Qwen, Llama, Gemma, Mistral,
Phi) is usable by setting ``OLLAMA_MODEL``.
"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator, Optional

from ai.common import settings

from .base import ChatTurn, LLMProvider, LLMResponse

logger = logging.getLogger("charlie.llm.ollama")


class OllamaProvider(LLMProvider):
    name = "ollama"
    generative = True

    def __init__(self, host: Optional[str] = None, model: Optional[str] = None):
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_model
        self._available: Optional[bool] = None
        self._checked_at = 0.0
        self._models: list[str] = []

    # -- availability ------------------------------------------------------
    def is_available(self) -> bool:
        # Re-probe periodically: Ollama is often started after the backend.
        if self._available is not None and time.time() - self._checked_at < 30:
            return self._available

        self._checked_at = time.time()
        try:
            import httpx

            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{self.host}/api/tags")
                response.raise_for_status()
                data = response.json()
            self._models = [m.get("name", "") for m in data.get("models", [])]
            self._available = True
            if self._models and not any(
                m.split(":")[0] == self.model.split(":")[0] for m in self._models
            ):
                logger.warning(
                    "Ollama is running but '%s' is not pulled. Available: %s. "
                    "Run: ollama pull %s",
                    self.model,
                    ", ".join(self._models) or "none",
                    self.model,
                )
                self._available = False
        except Exception:
            self._available = False
        return self._available

    def installed_models(self) -> list[str]:
        self.is_available()
        return self._models

    # -- generation --------------------------------------------------------
    async def generate(self, messages: list[ChatTurn], **kwargs) -> LLMResponse:
        import httpx

        started = time.time()
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", settings.llm_temperature),
                "num_predict": kwargs.get("max_tokens", settings.llm_max_tokens),
                # Small models degenerate into near-identical sentences without
                # this — observed on qwen2.5:1.5b emitting ten paragraphs of
                # "The video shows the surgeon using the instruments to X and
                # preparing the surgical field." 1.15 breaks the loop without
                # flattening clinical vocabulary, which repeats legitimately:
                # an instrument gets named every time it acts.
                "repeat_penalty": kwargs.get("repeat_penalty", 1.15),
                # The default look-back of 64 tokens is shorter than one of
                # those paragraphs, so the repetition was invisible to it.
                "repeat_last_n": 256,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout_s) as client:
                response = await client.post(f"{self.host}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
            text = (data.get("message") or {}).get("content", "").strip()
            return LLMResponse(
                text=text,
                provider=self.name,
                model=payload["model"],
                latency_ms=round((time.time() - started) * 1000, 1),
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
            )
        except Exception as exc:
            logger.error("Ollama generation failed: %s", exc)
            self._available = False
            return LLMResponse(
                text="",
                provider=self.name,
                model=payload["model"],
                latency_ms=round((time.time() - started) * 1000, 1),
                error=str(exc),
            )

    async def stream(self, messages: list[ChatTurn], **kwargs) -> AsyncIterator[str]:
        import httpx

        payload = {
            "model": kwargs.get("model", self.model),
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", settings.llm_temperature),
                "num_predict": kwargs.get("max_tokens", settings.llm_max_tokens),
                # Small models degenerate into near-identical sentences without
                # this — observed on qwen2.5:1.5b emitting ten paragraphs of
                # "The video shows the surgeon using the instruments to X and
                # preparing the surgical field." 1.15 breaks the loop without
                # flattening clinical vocabulary, which repeats legitimately:
                # an instrument gets named every time it acts.
                "repeat_penalty": kwargs.get("repeat_penalty", 1.15),
                # The default look-back of 64 tokens is shorter than one of
                # those paragraphs, so the repetition was invisible to it.
                "repeat_last_n": 256,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout_s) as client:
                async with client.stream(
                    "POST", f"{self.host}/api/chat", json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        piece = (data.get("message") or {}).get("content", "")
                        if piece:
                            yield piece
                        if data.get("done"):
                            break
        except Exception as exc:
            logger.error("Ollama streaming failed: %s", exc)
            self._available = False
            # Surface the failure rather than ending the stream silently.
            yield f"\n\n_(Local model stream interrupted: {exc})_"

    def info(self) -> dict:
        base = super().info()
        base.update({"host": self.host, "installed_models": self._models})
        return base
