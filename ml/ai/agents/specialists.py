"""
The five specialists.

Each owns exactly one source of truth, and each is explicit about what that
source is. That matters more here than in most agent systems: a surgeon
glancing at a tile has to be able to tell instantly whether "common bile duct —
critical" came from a static teaching reference or from something a model
claims to see in the frame. Those are very different claims, and conflating
them is the failure mode this whole design exists to avoid.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from ai.common import settings
from ai.knowledge import get_ontology, knowledge_graph, risks_for
from ai.temporal_memory import memory_engine

from .base import AgentContext, AgentResponse, AgentTile, Intent, SpecialistAgent, TileItem

logger = logging.getLogger("charlie.agents")


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


# ---------------------------------------------------------------------------
# Anatomy Identification
# ---------------------------------------------------------------------------
class AnatomyAgent(SpecialistAgent):
    """
    Surfaces the structures at risk during the *current* phase.

    Risk is phase-dependent: the common bile duct is the structure that matters
    during hepatocystic dissection and largely irrelevant during port
    placement. A flat per-specialty list would bury the one entry worth
    reading, so the map is indexed by (specialty, phase) and the answer changes
    as the procedure progresses.

    The map is static educational reference content. This agent does **not**
    claim to have seen these structures in the frame, and every tile says so —
    a tile that looked like a live detection would be a fabricated observation.
    """

    name = "anatomy"
    intent = Intent.ANATOMY
    description = (
        "At-risk structures and danger zones for the current surgical phase, "
        "with the manoeuvre that conventionally protects each one."
    )

    def handle(self, context: AgentContext) -> AgentResponse:
        started = time.time()

        phase = context.detected_phase or self._phase_from_memory(context)
        risks = risks_for(context.specialty, phase)

        items = [
            TileItem(
                label=risk["structure"],
                detail=risk["why"],
                severity=risk["severity"],
            )
            for risk in risks
        ]

        headline = "critical" if any(i.severity == "critical" for i in items) else "caution"
        phase_label = phase or "this specialty (no phase detected)"

        tiles = [
            AgentTile(
                agent=self.name,
                title=f"At risk — {phase_label}",
                items=self._rank(items),
                severity=headline,
                provenance="Static teaching reference, indexed by specialty and phase.",
                footnote=(
                    "Not a live detection: nothing here was observed in the video "
                    "frame. Educational reference only."
                ),
            )
        ]

        # A second tile with the protective manoeuvre for the top structures —
        # naming a danger without naming the response is only half useful.
        protections = [
            TileItem(label=risk["structure"], detail=risk["protect"], severity="info")
            for risk in risks[:3]
        ]
        if protections:
            tiles.append(
                AgentTile(
                    agent=self.name,
                    title="Conventional protective steps",
                    items=protections,
                    severity="info",
                    provenance="Static teaching reference.",
                )
            )

        if risks:
            top = risks[0]
            spoken = (
                f"Highest risk in {phase_label}: the {top['structure']}. "
                f"{top['protect']}"
            )
        else:
            spoken = "No specific risk entries for this phase."

        return AgentResponse(
            intent=self.intent,
            agent=self.name,
            tiles=tiles,
            spoken=spoken,
            confidence=0.9 if phase else 0.5,
            latency_ms=(time.time() - started) * 1000,
        )

    @staticmethod
    def _phase_from_memory(context: AgentContext) -> Optional[str]:
        if not context.session_id or context.timestamp is None:
            return None
        event = memory_engine.at_time(context.session_id, float(context.timestamp))
        return event.phase if event else None


# ---------------------------------------------------------------------------
# Surgical phase
# ---------------------------------------------------------------------------
class PhaseAgent(SpecialistAgent):
    """Where the procedure is, and what normally follows."""

    name = "phase"
    intent = Intent.PHASE
    description = "Current operative phase, progress through the workflow, and what comes next."

    def handle(self, context: AgentContext) -> AgentResponse:
        started = time.time()

        phase = context.detected_phase
        confidence = context.phase_confidence
        source = "live inference"

        if not phase and context.session_id and context.timestamp is not None:
            event = memory_engine.at_time(context.session_id, float(context.timestamp))
            if event:
                phase, confidence, source = event.phase, event.confidence, "temporal memory"

        phases = knowledge_graph.phase_names(context.specialty)

        if not phase:
            tile = AgentTile(
                agent=self.name,
                title="Phase not determined",
                items=[
                    TileItem(
                        label="No phase signal",
                        detail=(
                            "Nothing is being analysed right now. Enable live "
                            "inference, or load and index a procedure."
                        ),
                        severity="watch",
                    ),
                    TileItem(
                        label=f"{context.specialty} workflow",
                        detail=" → ".join(phases[:6]) if phases else "no ontology",
                        severity="info",
                    ),
                ],
                severity="watch",
                provenance="No observation available.",
            )
            return AgentResponse(
                intent=self.intent, agent=self.name, tiles=[tile],
                spoken="I can't determine the phase — nothing is being analysed right now.",
                confidence=0.0, latency_ms=(time.time() - started) * 1000,
            )

        next_phase = knowledge_graph.next_phase(context.specialty, phase)
        progress = knowledge_graph.progress(context.specialty, phase)
        detail = next(
            (p["description"] for p in knowledge_graph.phases(context.specialty)
             if p["name"].lower() == phase.lower()),
            "",
        )

        items = [
            TileItem(
                label=phase,
                detail=detail,
                severity="info",
                confidence=confidence,
                timestamp=context.timestamp,
            )
        ]
        if next_phase:
            items.append(TileItem(label=f"Next: {next_phase}", severity="info"))
        if progress is not None:
            items.append(
                TileItem(
                    label=f"Step {round(progress * len(phases))} of {len(phases)}",
                    detail="Ordinal position in the phase sequence — phases differ in duration, so this is not elapsed time.",
                    severity="info",
                )
            )

        # Low confidence must be visible, not buried in a number.
        severity = "watch" if confidence < 0.6 else "info"
        footnote = (
            "Low confidence — treat as tentative." if confidence < 0.6 else ""
        )

        tile = AgentTile(
            agent=self.name, title="Current phase", items=items, severity=severity,
            provenance=f"From {source} ({confidence:.0%} confidence).",
            footnote=footnote,
        )

        spoken = f"{phase}, {confidence:.0%} confidence."
        if next_phase:
            spoken += f" Normally followed by {next_phase}."

        return AgentResponse(
            intent=self.intent, agent=self.name, tiles=[tile], spoken=spoken,
            confidence=confidence, latency_ms=(time.time() - started) * 1000,
        )


# ---------------------------------------------------------------------------
# Imaging
# ---------------------------------------------------------------------------
class ImagingAgent(SpecialistAgent):
    """Analyses the current frame: instruments, attention, similar moments."""

    name = "imaging"
    intent = Intent.IMAGING
    description = "Frame analysis — instrument presence, Grad-CAM attention, visually similar moments."

    def handle(self, context: AgentContext) -> AgentResponse:
        started = time.time()
        from ai.vision import frame_analyzer

        if not context.frame_data_url:
            tile = AgentTile(
                agent=self.name, title="No frame captured",
                items=[TileItem(
                    label="Nothing to analyse",
                    detail="Load a video or start the camera, then ask again.",
                    severity="watch",
                )],
                severity="watch", provenance="No input.",
            )
            return AgentResponse(
                intent=self.intent, agent=self.name, tiles=[tile],
                spoken="There's no frame to analyse.", confidence=0.0,
                latency_ms=(time.time() - started) * 1000,
            )

        import base64

        payload = context.frame_data_url
        if "," in payload:
            payload = payload.split(",", 1)[1]

        try:
            analysis = frame_analyzer.analyze_bytes(
                base64.b64decode(payload), specialty=context.specialty, want_gradcam=True
            )
        except Exception as exc:
            logger.error("Imaging agent failed: %s", exc)
            analysis = {"error": str(exc), "predictions_available": False}

        if not analysis.get("predictions_available"):
            tile = AgentTile(
                agent=self.name, title="Instrument detection unavailable",
                items=[TileItem(
                    label="No trained checkpoint",
                    detail=analysis.get("notice") or analysis.get("error", ""),
                    severity="watch",
                )],
                severity="watch",
                provenance="Reported honestly rather than guessed.",
                footnote="Train a model: python -m ai.training.train_tool_detection",
            )
            return AgentResponse(
                intent=self.intent, agent=self.name, tiles=[tile],
                spoken="Instrument detection isn't available — no trained model is loaded.",
                confidence=0.0, latency_ms=(time.time() - started) * 1000,
            )

        present = [t for t in analysis.get("tools", []) if t.get("present")]
        items = [
            TileItem(label=t["display"], severity="info", confidence=t["confidence"])
            for t in present[:6]
        ] or [TileItem(label="No instruments above threshold", severity="watch")]

        tile = AgentTile(
            agent=self.name, title="Instruments in view", items=items, severity="info",
            provenance=f"{analysis.get('model_name', 'model')} · {analysis.get('inference_ms', 0):.0f} ms",
        )

        spoken = (
            f"I can see {', '.join(t['display'] for t in present[:3])}."
            if present
            else "No instruments detected above the confidence threshold."
        )

        return AgentResponse(
            intent=self.intent, agent=self.name, tiles=[tile], spoken=spoken,
            confidence=max((t["confidence"] for t in present), default=0.0),
            latency_ms=(time.time() - started) * 1000,
        )


# ---------------------------------------------------------------------------
# Case context
# ---------------------------------------------------------------------------
class CaseContextAgent(SpecialistAgent):
    """
    Context about the *recording* being studied — never about a patient.

    A production OR assistant would query the record here. This platform is
    explicitly educational and holds no patient data, so this agent reports the
    session's own metadata and declines patient-specific questions outright
    rather than appearing to have a record it does not have.
    """

    name = "case_context"
    intent = Intent.CASE_CONTEXT
    description = "Metadata about the procedure recording under study: specialty, duration, indexed events."

    #: Patterns that mean the question is about a person, not a recording.
    _PATIENT_TERMS = (
        "patient", "his ", "her ", "allergy", "allergies", "medication", "meds",
        "history", "labs", "vitals", "bloods", "inr", "creatinine", "diagnosis",
        "chart", "record", "consent", "age", "weight", "bmi", "comorbid",
    )

    def handle(self, context: AgentContext) -> AgentResponse:
        started = time.time()
        query = context.query.lower()

        if any(term in query for term in self._PATIENT_TERMS):
            tile = AgentTile(
                agent=self.name,
                title="No patient data on this platform",
                items=[
                    TileItem(
                        label="Out of scope by design",
                        detail=(
                            "This is an educational and research platform. It holds "
                            "no patient records and is not connected to any clinical "
                            "system, so it cannot answer questions about an individual."
                        ),
                        severity="caution",
                    ),
                    TileItem(
                        label="What I can answer",
                        detail="The procedure recording, its specialty, workflow, timeline, and anatomy.",
                        severity="info",
                    ),
                ],
                severity="caution",
                provenance="Platform policy.",
            )
            return AgentResponse(
                intent=self.intent, agent=self.name, tiles=[tile],
                spoken=(
                    "I don't have patient data. This platform is for education and "
                    "research only."
                ),
                confidence=1.0, latency_ms=(time.time() - started) * 1000,
            )

        memory = memory_engine.get(context.session_id) if context.session_id else None
        ontology = get_ontology(context.specialty)

        if memory is None:
            items = [
                TileItem(
                    label="No procedure loaded",
                    detail="Retrieve a video or start the camera to open a session.",
                    severity="watch",
                ),
                TileItem(label="Specialty", detail=context.specialty, severity="info"),
                TileItem(
                    label="Workflow",
                    detail=" → ".join(p["name"] for p in ontology.get("phases", [])[:6]),
                    severity="info",
                ),
            ]
            spoken = "No procedure is loaded right now."
        else:
            items = [
                TileItem(label="Recording", detail=memory.title, severity="info"),
                TileItem(label="Specialty", detail=memory.specialty, severity="info"),
                TileItem(label="Duration", detail=_hms(memory.duration), severity="info"),
                TileItem(
                    label="Indexed events",
                    detail=f"{len(memory.events)} segments · status {memory.status}",
                    severity="info",
                ),
                TileItem(
                    label="Source",
                    detail=memory.source + (f" — {memory.source_url}" if memory.source_url else ""),
                    severity="info",
                ),
            ]
            if context.timestamp is not None:
                items.insert(0, TileItem(
                    label="Playhead", detail=_hms(context.timestamp), severity="info",
                ))
            spoken = (
                f"{memory.title}. {memory.specialty}, {_hms(memory.duration)}, "
                f"{len(memory.events)} indexed events."
            )

        tile = AgentTile(
            agent=self.name, title="Case context", items=items, severity="info",
            provenance="Session metadata held by this platform.",
            footnote="No patient-identifiable information is stored.",
        )
        return AgentResponse(
            intent=self.intent, agent=self.name, tiles=[tile], spoken=spoken,
            latency_ms=(time.time() - started) * 1000,
        )


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------
class DocumentationAgent(SpecialistAgent):
    """
    Drafts an operative summary from the indexed timeline.

    Every line is traceable to an indexed event with its timestamp and
    confidence. It is a *draft from observations*, never an operative note: a
    generated note that reads as authoritative is a genuinely dangerous
    artefact, so the output is labelled as an unverified draft throughout.
    """

    name = "documentation"
    intent = Intent.DOCUMENTATION
    description = "Chronological summary of the indexed procedure, as an unverified draft."

    def handle(self, context: AgentContext) -> AgentResponse:
        started = time.time()

        memory = memory_engine.get(context.session_id) if context.session_id else None
        if memory is None or not memory.events:
            tile = AgentTile(
                agent=self.name, title="Nothing indexed yet",
                items=[TileItem(
                    label="No timeline",
                    detail="Load a procedure and let indexing finish, then ask again.",
                    severity="watch",
                )],
                severity="watch", provenance="No observations.",
            )
            return AgentResponse(
                intent=self.intent, agent=self.name, tiles=[tile],
                spoken="There's no indexed timeline to summarise yet.",
                confidence=0.0, latency_ms=(time.time() - started) * 1000,
            )

        items = [
            TileItem(
                label=f"{_hms(event.start)} — {event.label()}",
                detail=(
                    f"{_hms(event.duration)} · "
                    + (", ".join(event.tools[:3]) if event.tools else "no instruments recorded")
                ),
                severity="info",
                timestamp=event.start,
                confidence=event.confidence,
            )
            for event in memory.events[:12]
        ]

        low_confidence = sum(1 for e in memory.events if e.confidence < 0.6)
        footnote = "Unverified draft generated from model observations — review every line before use."
        if low_confidence:
            footnote += f" {low_confidence} of {len(memory.events)} segments are low-confidence."

        tile = AgentTile(
            agent=self.name,
            title=f"Procedure summary — {memory.title}",
            items=items,
            severity="info",
            provenance=f"{len(memory.events)} indexed segments over {_hms(memory.duration)}.",
            footnote=footnote,
        )

        phases = [e.phase for e in memory.events if e.phase]
        spoken = (
            f"{len(memory.events)} segments over {_hms(memory.duration)}, "
            f"from {phases[0] if phases else 'start'} to "
            f"{phases[-1] if phases else 'end'}. This is an unverified draft."
        )

        return AgentResponse(
            intent=self.intent, agent=self.name, tiles=[tile], spoken=spoken,
            confidence=sum(e.confidence for e in memory.events) / len(memory.events),
            latency_ms=(time.time() - started) * 1000,
        )
