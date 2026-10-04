"""
Speech-to-text.

Preference order:

1. ``faster-whisper`` — CTranslate2 Whisper, several times faster than the
   reference implementation and comfortable on CPU.
2. ``openai-whisper`` — the reference implementation.
3. Nothing server-side — the browser's Web Speech API handles transcription
   client-side instead. The endpoint reports this so the frontend knows to use
   its own recogniser rather than showing an error.

English is the default (``CHARLIE_STT_LANGUAGE=en``) as requested; Whisper is
multilingual, so setting that variable to another code, or to empty for
auto-detection, is all that is needed for other languages.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from typing import Optional

from ai.common import optional_import, settings

logger = logging.getLogger("charlie.voice.stt")


class SpeechToText:
    """Transcribes audio bytes to text using whatever local engine is present."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.stt_model
        self.backend = "browser"
        self._model = None
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        faster = optional_import("faster_whisper")
        if faster is not None:
            try:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.model_name, device="auto", compute_type="int8"
                )
                self.backend = f"faster-whisper:{self.model_name}"
                logger.info("STT ready: %s", self.backend)
                return
            except Exception as exc:
                logger.warning("faster-whisper load failed (%s)", exc)

        whisper = optional_import("whisper")
        if whisper is not None:
            try:
                self._model = whisper.load_model(self.model_name.replace(".en", ""))
                self.backend = f"whisper:{self.model_name}"
                logger.info("STT ready: %s", self.backend)
                return
            except Exception as exc:
                logger.warning("whisper load failed (%s)", exc)

        logger.info(
            "No local STT engine — the browser Web Speech API will handle "
            "transcription. Install faster-whisper for server-side STT."
        )

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def transcribe(
        self, audio: bytes, filename: str = "audio.webm", language: Optional[str] = None
    ) -> dict:
        """
        Transcribe encoded audio (webm/ogg/wav/mp3).

        Returns ``{text, language, duration, backend, segments}``. When no
        engine is installed, ``ok`` is False and ``fallback`` is ``"browser"``.
        """
        if self._model is None:
            return {
                "ok": False,
                "fallback": "browser",
                "backend": self.backend,
                "text": "",
                "message": (
                    "No server-side speech engine installed. The browser's "
                    "Web Speech API is used instead. For fully offline STT: "
                    "pip install faster-whisper"
                ),
            }

        language = language or settings.stt_language or None
        suffix = os.path.splitext(filename)[1] or ".webm"
        started = time.time()

        # Both engines read from a path; a temp file is the reliable route for
        # browser-recorded webm/opus.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(audio)
            path = handle.name

        try:
            with self._lock:
                if self.backend.startswith("faster-whisper"):
                    segments, info = self._model.transcribe(
                        path, language=language, vad_filter=True
                    )
                    parts = [
                        {
                            "start": round(s.start, 2),
                            "end": round(s.end, 2),
                            "text": s.text.strip(),
                        }
                        for s in segments
                    ]
                    text = " ".join(p["text"] for p in parts).strip()
                    detected = getattr(info, "language", language)
                    duration = getattr(info, "duration", 0.0)
                else:
                    result = self._model.transcribe(path, language=language)
                    parts = [
                        {
                            "start": round(s.get("start", 0), 2),
                            "end": round(s.get("end", 0), 2),
                            "text": s.get("text", "").strip(),
                        }
                        for s in result.get("segments", [])
                    ]
                    text = (result.get("text") or "").strip()
                    detected = result.get("language", language)
                    duration = parts[-1]["end"] if parts else 0.0

            return {
                "ok": True,
                "text": text,
                "language": detected,
                "duration": round(float(duration or 0), 2),
                "latency_ms": round((time.time() - started) * 1000, 1),
                "backend": self.backend,
                "segments": parts,
            }
        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            return {"ok": False, "text": "", "backend": self.backend, "error": str(exc)}
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def info(self) -> dict:
        return {
            "backend": self.backend,
            "available": self.is_available,
            "model": self.model_name,
            "language": settings.stt_language,
            "fallback": None if self.is_available else "browser Web Speech API",
        }


_INSTANCE: Optional[SpeechToText] = None
_LOCK = threading.Lock()


def get_stt() -> SpeechToText:
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = SpeechToText()
    return _INSTANCE
