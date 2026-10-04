"""
Gemini multimodal path — optional.

Two modes, because the Live API and the ordinary multimodal endpoint have
genuinely different reliability profiles:

``multimodal`` *(default)*
    One request carrying the transcript plus the current frame as an image.
    Works with any vision-capable Gemini model and any recent SDK version.
    This is the path the platform actually relies on.

``live``
    The bidirectional Live API — native audio in, audio out, video frames
    streamed at ~1 fps. Lower latency and no separate STT/TTS round trip, but
    it needs a preview model, an SDK that exposes ``client.aio.live``, and a
    persistent socket. Detected at runtime and reported honestly; when it is
    not available the platform says so rather than silently degrading in a way
    that looks like the feature is working.

Neither mode is required. The platform's default remains fully local — Ollama
plus Whisper plus Piper — and this exists because a `GEMINI_API_KEY` is already
present in the AI Studio environment this project came from.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, AsyncIterator, Optional

from ai.common import optional_import, settings

logger = logging.getLogger("charlie.llm.gemini_live")

#: Frames per second sent in live mode. The surgical field changes slowly
#: relative to video frame rate, and 1 fps is enough context for the model
#: while keeping bandwidth and token cost sane over a long procedure.
LIVE_VIDEO_FPS = 1.0

DEFAULT_LIVE_MODEL = "gemini-2.0-flash-live-001"
DEFAULT_VISION_MODEL = "gemini-2.0-flash"


class GeminiMultimodal:
    """Single-turn multimodal reasoning over a transcript plus a video frame."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model = model or settings.gemini_model or DEFAULT_VISION_MODEL
        self._client = None

    def is_available(self) -> bool:
        if not self.api_key or self.api_key.startswith("MY_"):
            return False
        return optional_import("google_genai") is not None

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def supports_live(self) -> bool:
        """Whether this SDK build exposes the bidirectional Live API."""
        if not self.is_available():
            return False
        try:
            client = self._get_client()
            return hasattr(getattr(client, "aio", None), "live")
        except Exception:
            return False

    async def analyze(
        self,
        prompt: str,
        frame_data_url: Optional[str] = None,
        system: str = "",
        **kwargs,
    ) -> dict:
        """
        Answer a question about the current frame.

        Returns ``{ok, text, model, latency_ms}`` or ``{ok: False, error}``.
        """
        if not self.is_available():
            return {
                "ok": False,
                "error": "No GEMINI_API_KEY set, or google-genai is not installed.",
            }

        started = time.time()
        try:
            from google.genai import types

            parts: list[Any] = [{"text": prompt}]

            if frame_data_url:
                payload = frame_data_url
                mime = "image/jpeg"
                if payload.startswith("data:"):
                    header, payload = payload.split(",", 1)
                    if ";" in header and ":" in header:
                        mime = header.split(":", 1)[1].split(";", 1)[0] or mime
                parts.insert(
                    0,
                    {"inline_data": {"mime_type": mime, "data": base64.b64decode(payload)}},
                )

            client = self._get_client()
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=[{"role": "user", "parts": parts}],
                config=types.GenerateContentConfig(
                    system_instruction=system or None,
                    temperature=kwargs.get("temperature", settings.llm_temperature),
                    max_output_tokens=kwargs.get("max_tokens", 600),
                ),
            )
            return {
                "ok": True,
                "text": (response.text or "").strip(),
                "model": self.model,
                "latency_ms": round((time.time() - started) * 1000, 1),
            }
        except Exception as exc:
            logger.error("Gemini multimodal call failed: %s", exc)
            return {
                "ok": False,
                "error": str(exc),
                "latency_ms": round((time.time() - started) * 1000, 1),
            }

    async def live_session(
        self, system: str = "", model: Optional[str] = None
    ) -> AsyncIterator[Any]:
        """
        Open a bidirectional Live session.

        Yields the connected session so a caller can push audio and frames.
        Raises ``RuntimeError`` when the SDK build has no Live support, rather
        than falling back silently — a caller asking for Live needs to know it
        did not get it.
        """
        if not self.supports_live():
            raise RuntimeError(
                "This google-genai build does not expose the Live API. "
                "Upgrade with: pip install -U google-genai — or use multimodal mode."
            )

        from google.genai import types

        client = self._get_client()
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=system or None,
        )
        async with client.aio.live.connect(
            model=model or DEFAULT_LIVE_MODEL, config=config
        ) as session:
            yield session

    def info(self) -> dict:
        available = self.is_available()
        return {
            "available": available,
            "mode": "live" if self.supports_live() else ("multimodal" if available else "unavailable"),
            "model": self.model,
            "live_model": DEFAULT_LIVE_MODEL,
            "video_fps": LIVE_VIDEO_FPS,
            "supports_live": self.supports_live(),
            "note": (
                None
                if available
                else "Set GEMINI_API_KEY and install google-genai to enable. "
                "Not required — the local pipeline handles vision and voice."
            ),
        }


_INSTANCE: Optional[GeminiMultimodal] = None


def get_gemini_multimodal() -> GeminiMultimodal:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = GeminiMultimodal()
    return _INSTANCE
