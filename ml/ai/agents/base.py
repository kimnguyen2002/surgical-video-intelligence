"""
Shared contract for specialist agents.

The tile is the unit of output. It exists because the target surface is a
surgical display seen at a glance by someone whose hands are on the controls:
a paragraph is unreadable there, but a titled tile with three ranked lines is
not. Every agent therefore returns tiles plus one short spoken confirmation,
regardless of how it computed the answer.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class Intent(str, Enum):
    """What the surgeon is asking for."""

    ANATOMY = "anatomy"            # what is at risk / what am I looking at
    PHASE = "phase"                # where are we in the procedure
    IMAGING = "imaging"            # analyse this frame / show attention
    CASE_CONTEXT = "case_context"  # what case is this, how long, what's indexed
    DOCUMENTATION = "documentation"  # summarise / draft the record
    EDUCATION = "education"        # open-ended teaching question → chat
    UNKNOWN = "unknown"


#: Tile severity drives colour and ordering on the display.
SEVERITY_RANK = {"critical": 0, "caution": 1, "watch": 2, "info": 3}


@dataclass
class TileItem:
    label: str
    detail: str = ""
    severity: str = "info"
    #: Seconds into the procedure, when the item refers to a moment.
    timestamp: Optional[float] = None
    confidence: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentTile:
    """One card on the surgical display."""

    agent: str
    title: str
    items: list[TileItem] = field(default_factory=list)
    severity: str = "info"
    #: Where this came from — shown so a static reference is never mistaken
    #: for something the model observed in the frame.
    provenance: str = ""
    footnote: str = ""

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "title": self.title,
            "severity": self.severity,
            "provenance": self.provenance,
            "footnote": self.footnote,
            "items": [i.to_dict() for i in self.items],
        }


@dataclass
class AgentResponse:
    """What the orchestrator returns for one command."""

    intent: Intent
    agent: str
    tiles: list[AgentTile] = field(default_factory=list)
    #: One sentence, written to be spoken aloud. No markdown, no citations.
    spoken: str = ""
    #: Set when the answer needs the full conversational assistant instead.
    defer_to_chat: bool = False
    confidence: float = 1.0
    latency_ms: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "agent": self.agent,
            "tiles": [t.to_dict() for t in self.tiles],
            "spoken": self.spoken,
            "defer_to_chat": self.defer_to_chat,
            "confidence": round(self.confidence, 3),
            "latency_ms": round(self.latency_ms, 1),
            "created_at": self.created_at,
        }


@dataclass
class AgentContext:
    """Everything a specialist may need about the current moment."""

    query: str = ""
    specialty: str = "General Surgery"
    session_id: Optional[str] = None
    timestamp: Optional[float] = None
    detected_phase: Optional[str] = None
    phase_confidence: float = 0.0
    tools: list[str] = field(default_factory=list)
    educational_mode: str = "Resident"
    #: Base64 frame, when the caller captured one for this command.
    frame_data_url: Optional[str] = None


class SpecialistAgent:
    """Base class. Subclasses implement :meth:`handle`."""

    name = "specialist"
    intent = Intent.UNKNOWN
    description = ""

    def handle(self, context: AgentContext) -> AgentResponse:
        raise NotImplementedError

    def info(self) -> dict:
        return {
            "name": self.name,
            "intent": self.intent.value,
            "description": self.description,
        }

    @staticmethod
    def _rank(items: list[TileItem]) -> list[TileItem]:
        return sorted(items, key=lambda i: SEVERITY_RANK.get(i.severity, 9))
