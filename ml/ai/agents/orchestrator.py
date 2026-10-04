"""
Intent routing.

Classification is deliberately lexical rather than model-based. Three reasons:

* **Latency.** A voice command routed through an LLM classifier adds hundreds
  of milliseconds before any specialist starts working. Hands-free use in an OR
  is exactly where that is least acceptable.
* **Determinism.** "What's at risk here" must reach the anatomy agent every
  single time, not 97% of the time.
* **Offline.** Routing has to work on the pure-Python tier, where no model is
  available at all.

Anything the patterns do not confidently claim is deferred to the full
conversational assistant, which is the right place for open-ended teaching
questions. A confident wrong route is worse than an honest hand-off.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from .base import AgentContext, AgentResponse, AgentTile, Intent, TileItem
from .specialists import (
    AnatomyAgent,
    CaseContextAgent,
    DocumentationAgent,
    ImagingAgent,
    PhaseAgent,
)

logger = logging.getLogger("charlie.orchestrator")


#: Ordered: the first intent whose pattern matches wins. Anatomy is checked
#: before phase because "what's at risk in this step" is a risk question that
#: happens to mention a step.
INTENT_PATTERNS: list[tuple[Intent, re.Pattern]] = [
    (
        Intent.ANATOMY,
        re.compile(
            r"\b(at risk|danger|dangerous|risky|avoid|injure|injury|damage|"
            r"critical structure|structures?\b.*\b(risk|avoid|near)|"
            r"what.{0,12}(nearby|around|behind|beneath)|anatomy|anatomical|"
            r"landmark|safe plane|watch out|careful|hazard)\b",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.PHASE,
        re.compile(
            r"\b(what (phase|step|stage)|which (phase|step|stage)|current (phase|step)|"
            r"where are we|how far|progress|what.{0,10}next|next step|workflow|"
            r"what.{0,12}happening now)\b",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.IMAGING,
        re.compile(
            r"\b(what (tool|instrument)|which (tool|instrument)|instruments?\b|"
            r"tools?\b|analyse this frame|analyze this frame|what do you see|"
            r"heatmap|grad.?cam|attention|look at this|scan this)\b",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.DOCUMENTATION,
        re.compile(
            r"\b(summar(y|ise|ize)|recap|document|documentation|op note|"
            r"operative note|report|timeline so far|what have we done|log)\b",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.CASE_CONTEXT,
        # Stems take `\w*` rather than a trailing `\b`: "allerg" followed by a
        # word boundary can never match "allergies", which is the form people
        # actually say.
        re.compile(
            r"\b(?:what case|which case|case (?:details|info)|patient\w*|"
            r"histor(?:y|ies)|allerg\w*|medication\w*|\bmeds\b|comorbid\w*|"
            r"how long (?:have|has|is)|duration|what video|"
            r"what procedure|what am i watching)",
            re.IGNORECASE,
        ),
    ),
]

#: Open-ended teaching questions belong in the chat assistant, not a tile.
EDUCATION_PATTERN = re.compile(
    r"\b(why|explain|teach|tell me about|how does|how do you|what is the "
    r"difference|compare|evidence|literature|study|paper|technique for)\b",
    re.IGNORECASE,
)


class Orchestrator:
    """Routes a command to the specialist that owns the relevant data."""

    def __init__(self):
        self._agents = {
            Intent.ANATOMY: AnatomyAgent(),
            Intent.PHASE: PhaseAgent(),
            Intent.IMAGING: ImagingAgent(),
            Intent.CASE_CONTEXT: CaseContextAgent(),
            Intent.DOCUMENTATION: DocumentationAgent(),
        }

    # -- routing ----------------------------------------------------------
    def classify(self, query: str) -> tuple[Intent, float]:
        """Return the intent and how confident the match is."""
        text = (query or "").strip()
        if not text:
            return Intent.UNKNOWN, 0.0

        matches = [
            (intent, pattern.search(text))
            for intent, pattern in INTENT_PATTERNS
        ]
        hits = [(intent, m) for intent, m in matches if m]

        if not hits:
            if EDUCATION_PATTERN.search(text):
                return Intent.EDUCATION, 0.6
            return Intent.UNKNOWN, 0.0

        intent = hits[0][0]
        # Several distinct intents matching means the phrasing is ambiguous;
        # report that rather than pretending to a clean classification.
        confidence = 0.95 if len(hits) == 1 else 0.7

        # An explicit teaching cue ("why is the CBD at risk") wants an
        # explanation, not a tile — but only when it is not also asking for a
        # specific live reading.
        if EDUCATION_PATTERN.search(text) and intent in {Intent.ANATOMY, Intent.CASE_CONTEXT}:
            if not re.search(r"\b(now|here|current|this (step|phase|frame))\b", text, re.I):
                return Intent.EDUCATION, 0.65

        return intent, confidence

    def route(self, context: AgentContext) -> AgentResponse:
        """Classify and dispatch. Never raises — a failed agent returns a tile."""
        started = time.time()
        intent, confidence = self.classify(context.query)

        if intent in {Intent.EDUCATION, Intent.UNKNOWN}:
            return AgentResponse(
                intent=intent,
                agent="chat",
                defer_to_chat=True,
                spoken="",
                confidence=confidence,
                latency_ms=(time.time() - started) * 1000,
            )

        agent = self._agents[intent]
        try:
            response = agent.handle(context)
        except Exception as exc:
            logger.exception("Specialist '%s' failed", agent.name)
            return AgentResponse(
                intent=intent,
                agent=agent.name,
                tiles=[
                    AgentTile(
                        agent=agent.name,
                        title=f"{agent.name} agent failed",
                        items=[TileItem(label="Error", detail=str(exc)[:200], severity="caution")],
                        severity="caution",
                        provenance="Internal error.",
                    )
                ],
                spoken="That agent hit an error.",
                confidence=0.0,
                latency_ms=(time.time() - started) * 1000,
            )

        # Routing confidence bounds answer confidence: a shaky route cannot
        # produce a confident answer.
        response.confidence = min(response.confidence, confidence)
        response.latency_ms = (time.time() - started) * 1000
        return response

    def specialists(self) -> list[dict]:
        return [agent.info() for agent in self._agents.values()]

    def describe_routing(self) -> dict:
        """Shown in the console so the routing table is inspectable."""
        return {
            "strategy": "lexical intent patterns, ordered by specificity",
            "rationale": (
                "Deterministic and sub-millisecond. A voice command in an OR "
                "cannot wait on an LLM classifier, and 'what is at risk' must "
                "route identically every time."
            ),
            "specialists": self.specialists(),
            "fallback": "Unmatched or open-ended questions defer to the conversational assistant.",
        }


orchestrator = Orchestrator()
