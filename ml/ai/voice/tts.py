"""
Text-to-speech.

Preference order:

1. **Piper** — small, fast, fully offline neural TTS. Set ``CHARLIE_TTS_VOICE``
   to a downloaded ``.onnx`` voice model, or drop one into
   ``data/cache/piper/`` and it is picked up automatically.
2. **macOS ``say``** — present on every Mac, useful for local development.
3. **Browser SpeechSynthesis** — the endpoint reports ``fallback: "browser"``
   and the frontend speaks the text client-side.

Because tier 3 always exists, the voice assistant speaks in every environment;
the server tiers exist so a deployment can be fully offline and consistent
across clients.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from ai.common import settings

logger = logging.getLogger("charlie.voice.tts")


class TextToSpeech:
    """Synthesises speech, returning WAV bytes when a local engine exists."""

    def __init__(self, voice_model: Optional[str] = None):
        self.voice_model = voice_model or settings.tts_voice_model
        self.backend = "browser"
        self._piper_bin: Optional[str] = None
        self._lock = threading.Lock()
        self._detect()

    def _detect(self) -> None:
        piper = shutil.which("piper")
        if piper:
            model = self._find_voice_model()
            if model:
                self._piper_bin = piper
                self.voice_model = str(model)
                self.backend = f"piper:{Path(model).stem}"
                logger.info("TTS ready: %s", self.backend)
                return
            logger.warning(
                "piper is installed but no voice model was found. Download one "
                "from https://github.com/rhasspy/piper/releases and set "
                "CHARLIE_TTS_VOICE, or place the .onnx file in %s",
                settings.cache_dir / "piper",
            )

        if shutil.which("say"):
            self.backend = "macos-say"
            logger.info("TTS ready: macOS 'say'")
            return

        logger.info(
            "No local TTS engine — the browser SpeechSynthesis API will speak "
            "replies. For offline TTS: pip install piper-tts"
        )

    def _find_voice_model(self) -> Optional[Path]:
        if self.voice_model and Path(self.voice_model).exists():
            return Path(self.voice_model)
        voices_dir = settings.cache_dir / "piper"
        if voices_dir.exists():
            models = sorted(voices_dir.glob("*.onnx"))
            if models:
                return models[0]
        return None

    @property
    def is_available(self) -> bool:
        return self.backend != "browser"

    def synthesize(self, text: str) -> dict:
        """
        Synthesise speech.

        Returns ``{ok, audio (WAV bytes), backend}``, or ``ok=False`` with
        ``fallback="browser"`` when no server engine is present.
        """
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "No text supplied.", "backend": self.backend}

        if not self.is_available:
            return {
                "ok": False,
                "fallback": "browser",
                "backend": self.backend,
                "message": (
                    "No server-side TTS engine installed; the browser will "
                    "speak this reply. For offline TTS: pip install piper-tts"
                ),
            }

        # Speech synthesis of a whole essay is slow and rarely wanted; the
        # spoken reply is meant to be a summary.
        text = text[:2000]
        started = time.time()

        try:
            with self._lock:
                if self.backend.startswith("piper"):
                    audio = self._piper(text)
                else:
                    audio = self._macos_say(text)
            return {
                "ok": True,
                "audio": audio,
                "format": "wav",
                "backend": self.backend,
                "latency_ms": round((time.time() - started) * 1000, 1),
            }
        except Exception as exc:
            logger.error("Speech synthesis failed: %s", exc)
            return {
                "ok": False,
                "fallback": "browser",
                "backend": self.backend,
                "error": str(exc),
            }

    def _piper(self, text: str) -> bytes:
        assert self._piper_bin
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output = handle.name
        try:
            subprocess.run(
                [self._piper_bin, "--model", self.voice_model, "--output_file", output],
                input=text.encode("utf-8"),
                check=True,
                capture_output=True,
                timeout=60,
            )
            return Path(output).read_bytes()
        finally:
            Path(output).unlink(missing_ok=True)

    def _macos_say(self, text: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as handle:
            aiff = handle.name
        wav = aiff.replace(".aiff", ".wav")
        try:
            subprocess.run(["say", "-o", aiff, text], check=True, timeout=60)
            # afconvert ships with macOS, so the browser gets standard WAV.
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", aiff, wav],
                check=True,
                timeout=60,
            )
            return Path(wav).read_bytes()
        finally:
            Path(aiff).unlink(missing_ok=True)
            Path(wav).unlink(missing_ok=True)

    def info(self) -> dict:
        return {
            "backend": self.backend,
            "available": self.is_available,
            "voice_model": self.voice_model or None,
            "language": settings.tts_language,
            "fallback": None if self.is_available else "browser SpeechSynthesis",
        }


_INSTANCE: Optional[TextToSpeech] = None
_LOCK = threading.Lock()


def get_tts() -> TextToSpeech:
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = TextToSpeech()
    return _INSTANCE
