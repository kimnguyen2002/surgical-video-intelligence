"""
Deterministic answers from what the platform actually knows.

This is what the assistant says when no language model is connected — and it
runs *before* the model when one is, because for factual questions about the
video ("what instrument is in use?", "what's happening?") the dataset's own
annotations and the detector's output are better evidence than anything a 7B
model will produce from a prompt.

Why this module exists
----------------------
The previous fallback ranked sentences from the whole system prompt against the
question. The system prompt contains the safety constraints and the audience
calibration, so asking "what instrument is in use right now?" returned:

    - Never use profanity, slurs, demeaning language, or crude humour…
    - Focus on operative workflow, the decision points within each phase…

It was quoting its own instructions back as if they were findings. Nothing in
that output is an answer, and the failure looks like a broken chatbot because
it is one.

The fix is not a better ranking function. It is to answer from **structured
facts** — recorded annotations, detector output, the procedure timeline — and
to fall back to text extraction only over *retrieved evidence*, never over the
prompt's own instruction layers.

Provenance is carried into the wording. "The dataset records a needle driver"
and "the detector reports a needle driver at 71% confidence" are different
claims, and the sentence the user reads should make that obvious without them
having to check a badge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

# Question intents. **Order is significant** — the first match wins, so the
# more specific pattern must come first.
#
# Two orderings here are load-bearing and were both wrong on the first pass:
#
#   * `summary` must precede `task`, because "summarise this procedure"
#     contains "procedure", which the task pattern matches.
#   * `count` must precede `instrument`, because "how many instruments" names
#     an instrument.
#
# Stems are matched with `\w*` rather than a closing `\b`: `\b(summar)\b`
# does not match "summarise", since the boundary assertion fails mid-word.
_INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("when", re.compile(
        r"\b(when|at what (?:time|point)|how long|timestamps?|what time)\b", re.I)),
    ("summary", re.compile(
        r"\b(summar\w*|overview|recap|walk me through|describe (?:the|this)\s+"
        r"(?:whole|entire|full|video|procedure|operation|case))\b", re.I)),
    ("count", re.compile(r"\b(how many|how much|count|number of)\b", re.I)),
    ("risk", re.compile(
        r"\b(risks?|danger\w*|at risk|avoid|careful|complications?|injur\w*)\b", re.I)),
    # "What is on screen?" / "what can you see?" names no instrument, but it is
    # one of the most common ways a user asks this question — and when boxes
    # exist they are the best possible answer to it.
    ("instrument", re.compile(
        r"\b(instruments?|tools?|devices?|forceps|scissors|drivers?|staplers?|"
        r"graspers?|sealers?|cautery|clip applier|retractors?"
        r"|on screen|on-screen|visible|can you see|in (?:the )?frame)\b", re.I)),
    ("task", re.compile(
        r"\b(tasks?|steps?|phase|stage|procedure|doing|happening|going on|"
        r"what.*(?:now|currently))\b", re.I)),
    ("list", re.compile(r"\b(list|which|what)\b.*\b(used|present|appear\w*|seen)\b", re.I)),
]


@dataclass
class VideoFacts:
    """Everything structured the answerer is allowed to assert."""

    #: `ai.datasets.groundtruth.FrameTruth.to_dict()`, or None.
    ground_truth: Optional[dict] = None
    #: Detector output for the current frame — `provenance: "predicted"`.
    detections: list[dict] = field(default_factory=list)
    #: Whole-video annotated timeline (`TimelineEntry.to_dict()`).
    timeline: list[dict] = field(default_factory=list)
    #: Retrieved RAG passages (objects with `.excerpt` / `.citation()`).
    passages: list = field(default_factory=list)
    timestamp: Optional[float] = None
    title: Optional[str] = None
    specialty: str = "General Surgery"
    #: Risk entries from the knowledge graph for the current phase.
    risks: list[dict] = field(default_factory=list)

    @property
    def has_any(self) -> bool:
        return bool(
            self.ground_truth and self.ground_truth.get("has_ground_truth")
        ) or bool(self.detections) or bool(self.timeline)


def classify_intent(question: str) -> str:
    for name, pattern in _INTENT_PATTERNS:
        if pattern.search(question or ""):
            return name
    return "general"


def hms(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


class GroundedAnswerer:
    """Composes a factual answer, or returns ``None`` if it cannot."""

    def answer(self, question: str, facts: VideoFacts) -> Optional[str]:
        if not facts.has_any:
            return None

        intent = classify_intent(question)
        handler = {
            "instrument": self._instruments,
            "task": self._task,
            "when": self._when,
            "count": self._count,
            "summary": self._summary,
            "list": self._instruments,
            "risk": self._risk,
        }.get(intent)

        if handler is None:
            # For an unclassified question, still lead with the current state
            # if there is one — it is almost always the context the user means.
            return self._task(question, facts)
        return handler(question, facts)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _recorded_tools(facts: VideoFacts) -> list[str]:
        gt = facts.ground_truth or {}
        if not gt.get("has_ground_truth"):
            return []
        return list(gt.get("tools_display") or gt.get("tools") or [])

    @staticmethod
    def _boxes(facts: VideoFacts) -> list[dict]:
        return list((facts.ground_truth or {}).get("boxes") or [])

    def _at(self, facts: VideoFacts) -> str:
        ts = facts.timestamp
        if ts is None:
            ts = (facts.ground_truth or {}).get("timestamp")
        return f"At **{hms(ts)}**" if ts is not None else "Right now"

    # -- intents ----------------------------------------------------------

    def _instruments(self, question: str, facts: VideoFacts) -> Optional[str]:
        lines: list[str] = []
        recorded = self._recorded_tools(facts)
        boxes = self._boxes(facts)

        if boxes:
            named = sorted({str(b.get("display") or b.get("class")) for b in boxes})
            lines.append(
                f"{self._at(facts)}, the dataset's frame annotations show "
                f"**{_join(named)}** on screen "
                f"({len(boxes)} labelled box{'es' if len(boxes) != 1 else ''})."
            )
        elif recorded:
            lines.append(
                f"{self._at(facts)}, the dataset records **{_join(recorded)}** "
                "mounted on the robot arms."
            )
            lines.append(
                "That comes from the installation log rather than from vision, "
                "so it means *installed* — an instrument stays on the list while "
                "it is off-screen or occluded."
            )

        if facts.detections:
            described = ", ".join(
                f"{d.get('display') or d.get('class')} ({float(d.get('confidence', 0)):.0%})"
                for d in facts.detections[:5]
            )
            lines.append(f"The detector currently reports: {described}.")
            if recorded or boxes:
                lines.append(
                    "_The detector is a small model trained on five clips; where "
                    "it disagrees with the annotations above, trust the annotations._"
                )

        if not lines:
            return None
        return "\n\n".join(lines)

    def _task(self, question: str, facts: VideoFacts) -> Optional[str]:
        gt = facts.ground_truth or {}
        task = gt.get("task_display") or gt.get("task")
        lines: list[str] = []

        if task:
            lines.append(
                f"{self._at(facts)}, the annotated surgical task is **{task}**."
            )
        elif gt.get("has_ground_truth"):
            lines.append(
                f"{self._at(facts)}, no surgical task is annotated — this falls "
                "between labelled segments."
            )

        tools = self._recorded_tools(facts)
        if tools:
            lines.append(f"Instruments in play: {_join(tools)}.")

        nearby = self._nearby_events(facts, window=180)
        if nearby:
            lines.append("Around this point:")
            lines.extend(
                f"- **{hms(e['start'])}** — {e.get('display') or e.get('label')}"
                for e in nearby[:5]
            )

        if not lines:
            return None
        return "\n\n".join(lines)

    def _when(self, question: str, facts: VideoFacts) -> Optional[str]:
        if not facts.timeline:
            return None

        terms = {w for w in re.findall(r"[a-z]+", (question or "").lower()) if len(w) > 3}
        matches = []
        for event in facts.timeline:
            label = f"{event.get('display', '')} {event.get('label', '')}".lower()
            if any(t in label for t in terms):
                matches.append(event)

        if not matches:
            return None

        matches.sort(key=lambda e: e.get("start", 0))
        head = matches[0]
        lines = [
            f"**{head.get('display') or head.get('label')}** first appears at "
            f"**{hms(head.get('start', 0))}** and runs to "
            f"**{hms(head.get('end', 0))}**."
        ]
        if len(matches) > 1:
            lines.append(f"It occurs {len(matches)} times in this recording:")
            lines.extend(
                f"- **{hms(e.get('start', 0))} – {hms(e.get('end', 0))}** "
                f"({_duration(e)})"
                for e in matches[:8]
            )
        lines.append("_Times are from the dataset's own annotations._")
        return "\n\n".join(lines)

    def _count(self, question: str, facts: VideoFacts) -> Optional[str]:
        boxes = self._boxes(facts)
        if boxes:
            per_class: dict[str, int] = {}
            for box in boxes:
                key = str(box.get("display") or box.get("class"))
                per_class[key] = per_class.get(key, 0) + 1
            listed = ", ".join(f"{count}× {name}" for name, count in per_class.items())
            return (
                f"{self._at(facts)} there are **{len(boxes)}** annotated "
                f"instrument{'s' if len(boxes) != 1 else ''} in frame: {listed}."
            )
        tools = self._recorded_tools(facts)
        if tools:
            return (
                f"{self._at(facts)} the dataset records **{len(tools)}** "
                f"instrument{'s' if len(tools) != 1 else ''} installed: {_join(tools)}."
            )
        return None

    def _summary(self, question: str, facts: VideoFacts) -> Optional[str]:
        if not facts.timeline:
            return self._task(question, facts)

        tools: dict[str, float] = {}
        tasks: dict[str, float] = {}
        for event in facts.timeline:
            bucket = tools if event.get("kind") in {"tool", "box"} else tasks
            name = event.get("display") or event.get("label") or "?"
            bucket[name] = bucket.get(name, 0.0) + float(event.get("duration", 0) or 0)

        lines = []
        if facts.title:
            lines.append(f"**{facts.title}**")
        lines.append(
            f"The recording carries **{len(facts.timeline)}** annotated events."
        )
        if tasks:
            ordered = sorted(tasks.items(), key=lambda kv: -kv[1])[:5]
            lines.append("Surgical tasks by time:")
            lines.extend(f"- {name} — {hms(secs)}" for name, secs in ordered)
        if tools:
            ordered = sorted(tools.items(), key=lambda kv: -kv[1])[:6]
            lines.append("Instruments by time in use:")
            lines.extend(f"- {name} — {hms(secs)}" for name, secs in ordered)
        lines.append("_All figures read from the dataset's annotations._")
        return "\n\n".join(lines)

    def _risk(self, question: str, facts: VideoFacts) -> Optional[str]:
        if not facts.risks:
            return None
        gt = facts.ground_truth or {}
        task = gt.get("task_display") or gt.get("task") or "this phase"

        # Name the specialty in the answer itself, not just in a badge.
        #
        # These entries come from the knowledge graph for whichever specialty
        # is currently selected, which defaults to General Surgery. That
        # produced "common bile duct" and "cystic artery" — cholecystectomy
        # structures — for a SurgVU case whose annotated tasks are uterine horn
        # and rectal artery manipulation. Anatomy from the wrong operation
        # presented as "at risk here" is the most dangerous thing this
        # assistant could say, so the operation it belongs to is stated up
        # front where a reader cannot miss it.
        lines = [
            f"Structures conventionally at risk during **{task}**, "
            f"for **{facts.specialty}** — the specialty currently selected:"
        ]
        for risk in facts.risks[:6]:
            name = risk.get("structure") or risk.get("name") or "?"
            note = risk.get("note") or risk.get("why") or ""
            lines.append(f"- **{name}**{f' — {note}' if note else ''}")
        lines.append(
            "_Reference anatomy for this phase from the specialty knowledge "
            "graph — **not** detections, and not derived from this video. "
            "Nothing here was localised in the current frame. If the list does "
            "not match what you are watching, change the specialty selector._"
        )
        return "\n\n".join(lines)

    # -- shared -----------------------------------------------------------

    @staticmethod
    def _nearby_events(facts: VideoFacts, window: float = 120.0) -> list[dict]:
        if facts.timestamp is None or not facts.timeline:
            return []
        now = float(facts.timestamp)
        near = [
            e for e in facts.timeline
            if abs(float(e.get("start", 0)) - now) <= window
            or (float(e.get("start", 0)) <= now <= float(e.get("end", 0)))
        ]
        near.sort(key=lambda e: abs(float(e.get("start", 0)) - now))
        return near


def _join(items: Iterable[str]) -> str:
    items = [str(i) for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _duration(event: dict) -> str:
    seconds = float(event.get("duration", 0) or 0)
    return f"{hms(seconds)} long" if seconds else "instant"


answerer = GroundedAnswerer()
