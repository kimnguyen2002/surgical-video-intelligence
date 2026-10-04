"""
Turning a stream of per-frame observations into a timeline of events.

Frames are not independent. The segmenter consumes observations in
chronological order and opens a new event when the procedure has *actually*
changed, using three signals:

1. **Embedding drift** — cosine distance from the running centroid of the
   current event exceeds a threshold (the scene changed).
2. **Label change** — the predicted phase differs from the current event's,
   confirmed over consecutive frames so a single noisy frame cannot split an
   event.
3. **Duration cap** — very long events are split so the timeline stays
   navigable.

The confirmation window is what keeps the timeline stable: without it, a
model that flickers between two phases on adjacent frames produces dozens of
one-frame events instead of one clean transition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ai.common import settings


@dataclass
class FrameObservation:
    """One sampled frame and everything the perception stack inferred from it."""

    timestamp: float
    embedding: list[float] = field(default_factory=list)
    phase: Optional[str] = None
    phase_confidence: float = 0.0
    tools: list[str] = field(default_factory=list)
    transcript: str = ""
    caption: str = ""


@dataclass
class _OpenEvent:
    start: float
    end: float
    phase: Optional[str]
    centroid: list[float]
    count: int
    confidences: list[float] = field(default_factory=list)
    tools: dict[str, int] = field(default_factory=dict)
    transcripts: list[str] = field(default_factory=list)


class EventSegmenter:
    """
    Online segmentation of a procedure into coherent events.

    >>> seg = EventSegmenter(drift_threshold=0.5, confirm_frames=1)
    >>> _ = seg.observe(FrameObservation(0.0, [1, 0], phase="Dissection"))
    >>> _ = seg.observe(FrameObservation(2.0, [0, 1], phase="Hemostasis"))
    >>> len(seg.finish())
    2
    """

    #: Split any event that runs longer than this (seconds).
    MAX_EVENT_SECONDS = 180.0

    def __init__(
        self,
        drift_threshold: Optional[float] = None,
        confirm_frames: int = 2,
        min_event_seconds: float = 1.0,
    ):
        self.drift_threshold = (
            drift_threshold
            if drift_threshold is not None
            else settings.memory_event_threshold
        )
        self.confirm_frames = max(1, confirm_frames)
        self.min_event_seconds = min_event_seconds

        self._current: Optional[_OpenEvent] = None
        self._closed: list[dict] = []
        self._pending_phase: Optional[str] = None
        self._pending_count = 0

    # -- streaming API ----------------------------------------------------
    def observe(self, obs: FrameObservation) -> Optional[dict]:
        """
        Feed one frame. Returns the event that just *closed*, if any.
        """
        if self._current is None:
            self._open(obs)
            return None

        closed = None
        if self._should_split(obs):
            closed = self._close(obs.timestamp)
            self._open(obs)
        else:
            self._extend(obs)
        return closed

    def finish(self) -> list[dict]:
        """Close the open event and return the full timeline."""
        if self._current is not None:
            self._close(self._current.end)
        return self._closed

    # -- decisions --------------------------------------------------------
    def _should_split(self, obs: FrameObservation) -> bool:
        cur = self._current
        assert cur is not None

        if obs.timestamp - cur.start >= self.MAX_EVENT_SECONDS:
            return True

        # A phase change must persist for `confirm_frames` frames before it is
        # believed — this is what suppresses single-frame prediction flicker.
        if obs.phase and obs.phase != cur.phase:
            if obs.phase == self._pending_phase:
                self._pending_count += 1
            else:
                self._pending_phase = obs.phase
                self._pending_count = 1
            if self._pending_count >= self.confirm_frames:
                self._pending_phase = None
                self._pending_count = 0
                return obs.timestamp - cur.start >= self.min_event_seconds
        else:
            self._pending_phase = None
            self._pending_count = 0

        # Visual drift from the event centroid.
        if obs.embedding and cur.centroid:
            distance = 1.0 - _cosine(obs.embedding, cur.centroid)
            if distance > self.drift_threshold:
                return obs.timestamp - cur.start >= self.min_event_seconds

        return False

    # -- event bookkeeping ------------------------------------------------
    def _open(self, obs: FrameObservation) -> None:
        self._current = _OpenEvent(
            start=obs.timestamp,
            end=obs.timestamp,
            phase=obs.phase,
            centroid=list(obs.embedding),
            count=1,
            confidences=[obs.phase_confidence] if obs.phase else [],
            tools={t: 1 for t in obs.tools},
            transcripts=[obs.transcript] if obs.transcript else [],
        )

    def _extend(self, obs: FrameObservation) -> None:
        cur = self._current
        assert cur is not None
        cur.end = obs.timestamp
        cur.count += 1
        if obs.phase:
            cur.confidences.append(obs.phase_confidence)
            if cur.phase is None:
                cur.phase = obs.phase
        for tool in obs.tools:
            cur.tools[tool] = cur.tools.get(tool, 0) + 1
        if obs.transcript:
            cur.transcripts.append(obs.transcript)
        if obs.embedding:
            if not cur.centroid:
                cur.centroid = list(obs.embedding)
            else:
                # Running mean, then re-normalise so distances stay comparable.
                n = cur.count
                cur.centroid = [
                    c + (v - c) / n for c, v in zip(cur.centroid, obs.embedding)
                ]
                cur.centroid = _unit(cur.centroid)

    def _close(self, end: float) -> dict:
        cur = self._current
        assert cur is not None
        confidence = (
            sum(cur.confidences) / len(cur.confidences) if cur.confidences else 0.0
        )
        # A tool counts as present in the event if it was seen in at least a
        # third of the event's frames — this filters detection noise.
        threshold = max(1, cur.count // 3)
        tools = sorted(
            [t for t, n in cur.tools.items() if n >= threshold],
            key=lambda t: -cur.tools[t],
        )
        event = {
            "start": round(cur.start, 2),
            "end": round(max(end, cur.end), 2),
            "duration": round(max(end, cur.end) - cur.start, 2),
            "phase": cur.phase,
            "confidence": round(confidence, 4),
            "frame_count": cur.count,
            "tools": tools,
            "embedding": cur.centroid,
            "transcript": " ".join(cur.transcripts).strip()[:2000],
        }
        self._closed.append(event)
        self._current = None
        return event


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _unit(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else v
