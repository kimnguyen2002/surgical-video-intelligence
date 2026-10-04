"""
Multi-agent orchestration.

A single prompt cannot answer "what's at risk here?", "what phase are we in?",
and "draft the op note" equally well. Each of those has a different source of
truth — the risk map, the temporal memory, the event log — so the orchestrator
classifies intent and routes to the specialist that owns that data.

Every specialist returns the same :class:`AgentTile` shape, so the surgical
display renders results uniformly and the voice layer can speak a one-line
confirmation without knowing which agent answered.
"""

from .base import AgentResponse, AgentTile, Intent, SpecialistAgent
from .orchestrator import Orchestrator, orchestrator
from .specialists import (
    AnatomyAgent,
    CaseContextAgent,
    DocumentationAgent,
    ImagingAgent,
    PhaseAgent,
)

__all__ = [
    "Intent",
    "AgentTile",
    "AgentResponse",
    "SpecialistAgent",
    "Orchestrator",
    "orchestrator",
    "AnatomyAgent",
    "PhaseAgent",
    "ImagingAgent",
    "CaseContextAgent",
    "DocumentationAgent",
]
