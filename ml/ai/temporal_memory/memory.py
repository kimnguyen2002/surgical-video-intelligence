"""
Temporal memory engine.

Each analysed procedure gets a :class:`ProcedureMemory`: a persistent,
chronological record of what happened, indexed two ways —

* **by time**, so the UI can render a navigable timeline and answer
  "what happened just before suturing?";
* **by meaning**, via a vector index over each event's visual centroid and
  its text description, so users can ask "show every bleeding event" and get
  back real video segments rather than a generated guess.

Retrieval returns the stored observation with its timestamp and confidence.
The engine never invents events — everything it returns was actually observed
and indexed, which is what keeps downstream explanations attributable.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from ai.common import settings
from ai.embeddings import get_frame_embedder, get_text_embedder
from ai.vectorstore import VectorStore

from .segmenter import EventSegmenter, FrameObservation

logger = logging.getLogger("charlie.memory")


@dataclass
class MemoryEvent:
    """A contiguous, semantically coherent segment of a procedure."""

    id: str
    session_id: str
    start: float
    end: float
    duration: float
    phase: Optional[str]
    confidence: float
    tools: list[str] = field(default_factory=list)
    frame_count: int = 0
    transcript: str = ""
    description: str = ""
    kind: str = "phase"  # phase | instrument | anatomy | event

    def to_dict(self) -> dict:
        return asdict(self)

    def label(self) -> str:
        return self.phase or self.description or "Observation"


@dataclass
class ProcedureMemory:
    """Everything the platform remembers about one analysed procedure."""

    session_id: str
    title: str
    specialty: str
    source: str  # youtube | upload | camera | rtsp
    source_url: str = ""
    duration: float = 0.0
    created_at: float = field(default_factory=time.time)
    events: list[MemoryEvent] = field(default_factory=list)
    status: str = "indexing"  # indexing | ready | error
    error: str = ""
    #: Which embedding tiers produced this memory — recorded so a researcher
    #: can tell whether two sessions are actually comparable.
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "specialty": self.specialty,
            "source": self.source,
            "source_url": self.source_url,
            "duration": self.duration,
            "created_at": self.created_at,
            "status": self.status,
            "error": self.error,
            "provenance": self.provenance,
            "event_count": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }

    def summary(self) -> dict:
        d = self.to_dict()
        d.pop("events")
        return d


class TemporalMemoryEngine:
    """Owns every :class:`ProcedureMemory` and its vector indices."""

    def __init__(self, directory: Optional[Path] = None):
        self.directory = Path(directory or settings.memory_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._sessions: dict[str, ProcedureMemory] = {}

        text_embedder = get_text_embedder()
        frame_embedder = get_frame_embedder()
        # Two indices: one over what an event *looked* like, one over what it
        # is *described* as. Visual search finds "scenes like this frame";
        # text search answers natural-language questions.
        self._visual_index = VectorStore("memory_visual", dim=frame_embedder.dim)
        self._text_index = VectorStore("memory_text", dim=text_embedder.dim)

        self._load_all()

    # -- session lifecycle -------------------------------------------------
    def create_session(
        self,
        title: str,
        specialty: str = "General Surgery",
        source: str = "upload",
        source_url: str = "",
        session_id: Optional[str] = None,
    ) -> ProcedureMemory:
        session_id = session_id or uuid.uuid4().hex[:12]
        memory = ProcedureMemory(
            session_id=session_id,
            title=title,
            specialty=specialty,
            source=source,
            source_url=source_url,
            provenance={
                "frame_embedder": get_frame_embedder().info(),
                "text_embedder": get_text_embedder().info(),
            },
        )
        with self._lock:
            self._sessions[session_id] = memory
            self._persist(memory)
        logger.info("Created memory session %s (%s)", session_id, title)
        return memory

    def get(self, session_id: str) -> Optional[ProcedureMemory]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[dict]:
        with self._lock:
            return sorted(
                (m.summary() for m in self._sessions.values()),
                key=lambda s: s["created_at"],
                reverse=True,
            )

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            memory = self._sessions.pop(session_id, None)
            if memory is None:
                return False
            ids = [e.id for e in memory.events]
            self._visual_index.delete(ids)
            self._text_index.delete(ids)
            path = self.directory / f"{session_id}.json"
            path.unlink(missing_ok=True)
        return True

    # -- ingestion ---------------------------------------------------------
    def ingest_observations(
        self,
        session_id: str,
        observations: list[FrameObservation],
        duration: Optional[float] = None,
    ) -> list[MemoryEvent]:
        """
        Segment a chronological batch of frame observations into events and
        index them. Returns the events created.
        """
        memory = self._sessions.get(session_id)
        if memory is None:
            raise KeyError(f"Unknown session: {session_id}")

        segmenter = EventSegmenter()
        for obs in sorted(observations, key=lambda o: o.timestamp):
            segmenter.observe(obs)
        raw_events = segmenter.finish()

        events: list[MemoryEvent] = []
        text_batch: list[tuple[str, list[float], dict, str]] = []
        visual_batch: list[tuple[str, list[float], dict, str]] = []
        text_embedder = get_text_embedder()

        for index, raw in enumerate(raw_events):
            event_id = f"{session_id}:{index:04d}"
            description = self._describe(raw, memory)
            event = MemoryEvent(
                id=event_id,
                session_id=session_id,
                start=raw["start"],
                end=raw["end"],
                duration=raw["duration"],
                phase=raw["phase"],
                confidence=raw["confidence"],
                tools=raw["tools"],
                frame_count=raw["frame_count"],
                transcript=raw["transcript"],
                description=description,
                kind="phase" if raw["phase"] else "event",
            )
            events.append(event)

            metadata = {
                "session_id": session_id,
                "start": event.start,
                "end": event.end,
                "phase": event.phase,
                "tools": event.tools,
                "specialty": memory.specialty,
                "confidence": event.confidence,
                "title": memory.title,
            }
            searchable = " ".join(
                filter(None, [description, event.transcript, " ".join(event.tools)])
            )
            text_batch.append(
                (event_id, text_embedder.encode_one(searchable), metadata, searchable)
            )
            if raw["embedding"]:
                visual_batch.append((event_id, raw["embedding"], metadata, description))

        with self._lock:
            memory.events = events
            memory.duration = duration or (events[-1].end if events else 0.0)
            memory.status = "ready"
            if text_batch:
                self._text_index.add_many(text_batch)
            if visual_batch:
                self._visual_index.add_many(visual_batch)
            self._persist(memory)

        logger.info(
            "Indexed %d events for session %s (%.1fs)",
            len(events),
            session_id,
            memory.duration,
        )
        return events

    def mark_error(self, session_id: str, message: str) -> None:
        memory = self._sessions.get(session_id)
        if memory:
            with self._lock:
                memory.status = "error"
                memory.error = message
                self._persist(memory)

    @staticmethod
    def _describe(raw: dict, memory: ProcedureMemory) -> str:
        """A short natural-language description used for text retrieval."""
        parts = []
        if raw["phase"]:
            parts.append(f"{raw['phase']} phase")
        if raw["tools"]:
            parts.append("instruments: " + ", ".join(raw["tools"]))
        parts.append(f"{memory.specialty}")
        parts.append(f"from {_hms(raw['start'])} to {_hms(raw['end'])}")
        return "; ".join(parts)

    # -- retrieval ---------------------------------------------------------
    def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 8,
    ) -> list[dict]:
        """Semantic search across indexed events."""
        vector = get_text_embedder().encode_one(query)
        where = {"session_id": session_id} if session_id else None
        results = self._text_index.search(vector, top_k=top_k, where=where)
        return [self._hydrate(r) for r in results]

    def search_by_frame(
        self,
        embedding: list[float],
        session_id: Optional[str] = None,
        top_k: int = 8,
    ) -> list[dict]:
        """Find moments that *look* like the supplied frame."""
        where = {"session_id": session_id} if session_id else None
        results = self._visual_index.search(embedding, top_k=top_k, where=where)
        return [self._hydrate(r) for r in results]

    def at_time(self, session_id: str, timestamp: float) -> Optional[MemoryEvent]:
        """The event covering a given moment."""
        memory = self._sessions.get(session_id)
        if not memory:
            return None
        for event in memory.events:
            if event.start <= timestamp <= event.end:
                return event
        return None

    def context_window(
        self, session_id: str, timestamp: float, before: int = 2, after: int = 2
    ) -> list[MemoryEvent]:
        """
        Events surrounding a moment — the basis for "what happened just
        before X?" questions.
        """
        memory = self._sessions.get(session_id)
        if not memory or not memory.events:
            return []
        index = 0
        for i, event in enumerate(memory.events):
            if event.start <= timestamp:
                index = i
        low = max(0, index - before)
        high = min(len(memory.events), index + after + 1)
        return memory.events[low:high]

    def _hydrate(self, result) -> dict:
        meta = result.metadata
        session_id = meta.get("session_id")
        memory = self._sessions.get(session_id) if session_id else None
        return {
            "event_id": result.id,
            "score": result.score,
            "session_id": session_id,
            "session_title": meta.get("title", ""),
            "start": meta.get("start", 0.0),
            "end": meta.get("end", 0.0),
            "start_hms": _hms(meta.get("start", 0.0)),
            "phase": meta.get("phase"),
            "tools": meta.get("tools", []),
            "confidence": meta.get("confidence", 0.0),
            "specialty": meta.get("specialty"),
            "text": result.text,
            "source_url": memory.source_url if memory else "",
        }

    def timeline(self, session_id: str) -> list[dict]:
        """The timeline in the shape the frontend renders."""
        memory = self._sessions.get(session_id)
        if not memory:
            return []
        out = []
        for event in memory.events:
            out.append(
                {
                    "id": event.id,
                    "timestamp": event.start,
                    "end": event.end,
                    "label": event.label(),
                    "type": event.kind,
                    "confidence": event.confidence,
                    "explanation": event.description,
                    "tools": event.tools,
                }
            )
        return out

    def stats(self) -> dict:
        with self._lock:
            total_events = sum(len(m.events) for m in self._sessions.values())
        return {
            "sessions": len(self._sessions),
            "events": total_events,
            "text_index": self._text_index.stats(),
            "visual_index": self._visual_index.stats(),
        }

    # -- persistence -------------------------------------------------------
    def _persist(self, memory: ProcedureMemory) -> None:
        path = self.directory / f"{memory.session_id}.json"
        try:
            path.write_text(
                json.dumps(memory.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Could not persist session %s: %s", memory.session_id, exc)

    def _load_all(self) -> None:
        for path in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping unreadable memory file %s: %s", path.name, exc)
                continue
            events = [MemoryEvent(**e) for e in data.get("events", [])]
            memory = ProcedureMemory(
                session_id=data["session_id"],
                title=data.get("title", "Untitled"),
                specialty=data.get("specialty", "General Surgery"),
                source=data.get("source", "upload"),
                source_url=data.get("source_url", ""),
                duration=data.get("duration", 0.0),
                created_at=data.get("created_at", time.time()),
                events=events,
                status=data.get("status", "ready"),
                error=data.get("error", ""),
                provenance=data.get("provenance", {}),
            )
            self._sessions[memory.session_id] = memory
        if self._sessions:
            logger.info("Restored %d memory sessions", len(self._sessions))


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


#: Process-wide engine used by the API layer.
memory_engine = TemporalMemoryEngine()
