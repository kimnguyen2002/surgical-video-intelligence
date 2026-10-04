"""
16 kHz PCM stream handling for the voice-command WebSocket.

The browser streams raw 16-bit little-endian mono PCM at 16 kHz — the rate
Whisper and every other speech model here expects, so no server-side resampling
is needed and the client sends roughly 32 KB/s instead of the ~1.4 MB/s of
44.1 kHz float audio.

The hard part is deciding when an utterance has *ended*. Fixed-length chunks cut
commands in half; waiting for the client to say so adds a round trip. So this
tracks short-term energy and closes an utterance after a run of quiet frames —
crude compared with a neural VAD, but it needs no model, runs in microseconds,
and degrades predictably in a noisy room because the noise floor is calibrated
from the opening frames rather than assumed.
"""

from __future__ import annotations

import array
import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger("charlie.voice.pcm")

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # int16
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


def rms(samples: array.array) -> float:
    """Root-mean-square amplitude, normalised to 0–1."""
    if not samples:
        return 0.0
    total = 0
    for sample in samples:
        total += sample * sample
    return math.sqrt(total / len(samples)) / 32768.0


def to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw PCM in a WAV container so any decoder can read it."""
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


@dataclass
class UtteranceBuffer:
    """
    Accumulates PCM and reports when a complete utterance is available.

    >>> buffer = UtteranceBuffer()
    >>> _ = buffer.feed(b"\\x00\\x00" * 480)
    >>> buffer.duration_s > 0
    True
    """

    #: Close the utterance after this much continuous quiet.
    silence_ms: int = 700
    #: Ignore utterances shorter than this — they are almost always noise.
    min_speech_ms: int = 350
    #: Hard cap so a stuck-open microphone cannot buffer without limit.
    max_utterance_s: float = 30.0
    #: Multiplier over the calibrated noise floor that counts as speech.
    speech_factor: float = 2.5

    _pcm: bytearray = field(default_factory=bytearray)
    _pending: bytearray = field(default_factory=bytearray)
    _noise_floor: float = 0.0
    _calibration_frames: int = 0
    _speech_ms: int = 0
    _silence_ms_seen: int = 0
    _in_speech: bool = False

    #: Frames used to estimate the room's noise floor before gating begins.
    CALIBRATION_FRAMES = 10

    @property
    def duration_s(self) -> float:
        return len(self._pcm) / (SAMPLE_RATE * SAMPLE_WIDTH)

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    def reset(self) -> None:
        self._pcm.clear()
        self._pending.clear()
        self._speech_ms = 0
        self._silence_ms_seen = 0
        self._in_speech = False

    def feed(self, chunk: bytes) -> bool:
        """
        Add PCM. Returns True when a complete utterance is ready to transcribe.
        """
        self._pending.extend(chunk)
        ready = False

        frame_bytes = FRAME_SAMPLES * SAMPLE_WIDTH
        while len(self._pending) >= frame_bytes:
            frame = bytes(self._pending[:frame_bytes])
            del self._pending[:frame_bytes]

            samples = array.array("h")
            samples.frombytes(frame)
            level = rms(samples)

            # Calibrate against the actual room rather than a fixed threshold —
            # a laparoscopic tower and a quiet office have very different floors.
            if self._calibration_frames < self.CALIBRATION_FRAMES:
                self._noise_floor = (
                    (self._noise_floor * self._calibration_frames + level)
                    / (self._calibration_frames + 1)
                )
                self._calibration_frames += 1
                continue

            threshold = max(self._noise_floor * self.speech_factor, 0.006)
            is_speech = level > threshold

            if is_speech:
                self._in_speech = True
                self._speech_ms += FRAME_MS
                self._silence_ms_seen = 0
                self._pcm.extend(frame)
            elif self._in_speech:
                # Keep trailing quiet: clipping the tail cuts word endings.
                self._silence_ms_seen += FRAME_MS
                self._pcm.extend(frame)
                if (
                    self._silence_ms_seen >= self.silence_ms
                    and self._speech_ms >= self.min_speech_ms
                ):
                    ready = True
                elif self._silence_ms_seen >= self.silence_ms:
                    # Long enough quiet but too little speech — noise, not a
                    # command. Drop it without a transcription round trip.
                    self.reset()

            if self.duration_s >= self.max_utterance_s:
                ready = self._speech_ms >= self.min_speech_ms
                if not ready:
                    self.reset()

        return ready

    def take(self) -> bytes:
        """Return the buffered utterance as WAV bytes and reset."""
        pcm = bytes(self._pcm)
        self.reset()
        return to_wav(pcm)

    def stats(self) -> dict:
        return {
            "duration_s": round(self.duration_s, 2),
            "speech_ms": self._speech_ms,
            "noise_floor": round(self._noise_floor, 5),
            "calibrated": self._calibration_frames >= self.CALIBRATION_FRAMES,
        }
