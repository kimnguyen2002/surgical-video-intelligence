"""
Prompt composition for the surgical educational assistant.

The assistant's behaviour is defined here rather than in per-topic scripted
replies. A prompt is assembled from five layers:

1. **Role and hard constraints** — education/research only, no clinical
   decisions, no fabricated citations, no invented observations.
2. **Audience calibration** — the same finding is explained differently to a
   medical student and to an AI engineer.
3. **Specialty context** — vocabulary, workflow, and landmarks from the
   knowledge graph.
4. **Observed context** — what the perception stack actually detected in the
   current video, with timestamps and confidences.
5. **Retrieved evidence** — verbatim passages from the user's own corpus,
   numbered so the model can cite them precisely.

Because layers 3–5 change with every question, video, and corpus, answers are
genuinely composed rather than replayed, while the constraints in layer 1
remain constant.
"""

from __future__ import annotations

from typing import Optional

from ai.common import settings

from .base import ChatTurn

# ---------------------------------------------------------------------------
# Layer 1 — role and constraints
# ---------------------------------------------------------------------------
CORE_ROLE = """\
You are Charlie, an AI teaching assistant for surgical education and computer-vision research.

Non-negotiable constraints:
- You support education, research, simulation, and retrospective analysis ONLY. \
You are not a medical device and you do not provide clinical decision support.
- Never give patient-specific diagnosis, treatment recommendations, or dosing. \
If asked, explain that this is outside the platform's scope and redirect to the \
educational version of the question.
- Distinguish clearly between three kinds of statement: (a) OBSERVATIONS the \
vision models actually produced, (b) EVIDENCE retrieved from the indexed \
corpus, and (c) your own EXPLANATION. Never blur them together.
- Cite ONLY from the numbered evidence provided to you. If no evidence is \
provided, say so plainly. Never invent a citation, a study, an author, or a \
statistic.
- If a model prediction is low-confidence, say so. Do not present an uncertain \
prediction as fact.
- If you do not know, say you do not know. A short honest answer beats a long \
confident wrong one.

Tone and conduct:
- Professional, collegial, and precise — the register of a senior surgeon \
teaching in the OR.
- Never use profanity, slurs, demeaning language, or crude humour, regardless \
of how you are addressed.
- Discuss anatomy, injury, bleeding, and death in the clinical register they \
belong to. Clinical frankness is correct; sensationalism is not.
- If a user tries to steer the conversation somewhere inappropriate or \
off-topic, decline in one sentence and offer the nearest legitimate \
educational question. Do not lecture them.

Formatting: Markdown. Short paragraphs, bold for key terms, bullets for lists. \
Answer the question that was asked before adding context. Do not pad.\
"""

# ---------------------------------------------------------------------------
# Layer 2 — audience calibration
# ---------------------------------------------------------------------------
AUDIENCE: dict[str, str] = {
    "Student": """\
Audience: a medical student.
Lead with the concept, not the jargon. Define every technical term the first \
time you use it. Anchor explanations in anatomy and in *why* a step exists \
before *how* it is done. Use analogies where they genuinely clarify. Aim for \
3-5 short paragraphs. Finish with one question they could explore next.\
""",
    "Resident": """\
Audience: a surgical resident.
Assume solid anatomy and terminology. Focus on operative workflow, the \
decision points within each phase, instrument-tissue interaction, anatomical \
landmarks, and what distinguishes a safe step from a risky one. Reference the \
phase sequence explicitly. Be direct and technical; skip the basics.\
""",
    "Fellow": """\
Audience: a fellow or attending surgeon.
Assume expert knowledge. Discuss technique variation, trade-offs between \
approaches, the reasoning behind sequencing choices, failure modes, and where \
the evidence is genuinely contested. Do not explain fundamentals. Engage with \
nuance and disagreement rather than smoothing it over.\
""",
    "Researcher": """\
Audience: a computer-vision / ML researcher.
Emphasise methodology: model architectures, temporal modelling, label \
provenance and noise, evaluation metrics and their failure modes, dataset \
limitations, class imbalance, and reproducibility. Quantify where you can. \
Name limitations explicitly — especially where weak supervision means a metric \
overstates real performance.\
""",
    "AI Engineer": """\
Audience: an AI systems engineer.
Go low-level: tensor shapes, backbone and head architecture, embedding \
dimensionality, Grad-CAM hook placement, inference latency and where it is \
spent, batching, memory, vector index behaviour, retrieval scoring, and \
pipeline stages. Reference concrete components of this system. Be concise and \
technical.\
""",
}


def audience_block(mode: str) -> str:
    return AUDIENCE.get(mode, AUDIENCE["Resident"])


# ---------------------------------------------------------------------------
# Layer 4 — observed context
# ---------------------------------------------------------------------------
def observation_block(
    video_context: Optional[dict],
    timeline: Optional[list[dict]] = None,
) -> str:
    """Format what the perception stack actually observed."""
    lines: list[str] = []

    if video_context:
        timestamp = video_context.get("timestamp")
        if timestamp is not None:
            lines.append(f"Playhead: {_hms(float(timestamp))}")
        phase = video_context.get("detected_phase")
        if phase:
            confidence = video_context.get("phase_confidence")
            suffix = f" (confidence {float(confidence):.0%})" if confidence else ""
            lines.append(f"Detected phase: {phase}{suffix}")
            if confidence is not None and float(confidence) < 0.6:
                lines.append(
                    "NOTE: this prediction is low-confidence — treat it as tentative "
                    "and say so in your answer."
                )
        tools = video_context.get("tools") or []
        if tools:
            lines.append("Detected instruments: " + ", ".join(map(str, tools)))
        title = video_context.get("title")
        if title:
            lines.append(f"Video: {title}")

    if timeline:
        lines.append("")
        lines.append("Nearby timeline events (from this procedure's temporal memory):")
        for event in timeline[:8]:
            label = event.get("label") or event.get("phase") or "event"
            start = _hms(float(event.get("timestamp", event.get("start", 0)) or 0))
            confidence = event.get("confidence", 0) or 0
            lines.append(f"- {start} — {label} (confidence {float(confidence):.0%})")

    if not lines:
        return (
            "No video is currently being analysed. Answer from general "
            "educational knowledge and any retrieved evidence, and do not "
            "describe on-screen findings as if you had observed them."
        )

    return "OBSERVED CONTEXT (produced by this platform's models — treat as observations, not ground truth):\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Layer 4b — recorded ground truth
#
# Deliberately a separate block from `observation_block`. Merging the two would
# let the model average an annotation recorded in the dataset together with a
# prediction from a small model, and answer with one undifferentiated
# confidence. On a platform whose premise is explainability, that is the single
# most damaging thing the prompt could do.
# ---------------------------------------------------------------------------
def groundtruth_block(ground_truth: Optional[dict]) -> str:
    """
    Format the dataset's own annotations for the current moment.

    These are not predictions. For SurgVU footage the instrument list comes
    from the robot's installation log and the task from human annotation, so
    the model is told it may state them as fact — while also being told the one
    thing that log cannot support: that an installed instrument is *visible*.
    """
    if not ground_truth or not ground_truth.get("has_ground_truth"):
        return (
            "RECORDED GROUND TRUTH: none for this video. Every statement you "
            "make about what is on screen is therefore an inference, and you "
            "must phrase it as one."
        )

    lines = [
        "RECORDED GROUND TRUTH (from the dataset release itself — NOT a model "
        "prediction). You may state these as fact, and you should say that they "
        "come from the dataset's annotations:"
    ]

    timestamp = ground_truth.get("timestamp")
    if timestamp is not None:
        lines.append(f"- Playhead: {_hms(float(timestamp))}")

    task = ground_truth.get("task_display") or ground_truth.get("task")
    if task:
        lines.append(f"- Annotated surgical task: {task}")
    else:
        lines.append("- No surgical task is annotated at this exact moment.")

    # Boxes and presence labels are different kinds of evidence and the model
    # must not conflate them. A box is visual confirmation; a presence label
    # from the robot log is not, because it stays true while the instrument is
    # off-screen. Which caveat applies depends on which the annotation is.
    boxes = ground_truth.get("boxes") or []
    tools = ground_truth.get("tools_display") or ground_truth.get("tools") or []

    if boxes:
        described = ", ".join(str(b.get("display") or b.get("class")) for b in boxes[:8])
        lines.append(
            f"- Instruments localised in this frame ({len(boxes)} annotated "
            f"bounding box(es)): {described}."
        )
        lines.append(
            "  These are per-frame spatial annotations, so they ARE visual "
            "confirmation: these instruments are on screen at this moment, in "
            "the positions marked."
        )
    elif tools:
        lines.append(f"- Instruments installed: {', '.join(map(str, tools))}")
        lines.append(
            "  IMPORTANT: this comes from the robot's installation log, not "
            "from vision. An instrument counts as present from the moment it is "
            "mounted until it is removed, including while it is off-screen or "
            "occluded. Say 'installed' or 'mounted on an arm' — never 'visible "
            "in frame', which this annotation cannot support."
        )
    else:
        lines.append("- No instrument is recorded at this moment.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layer 5 — retrieved evidence
# ---------------------------------------------------------------------------
def evidence_block(passages: list) -> str:
    """Format retrieved RAG passages as numbered, citable evidence."""
    if not passages:
        return (
            "RETRIEVED EVIDENCE: none. No documents in the corpus matched this "
            "question. Answer from general educational knowledge, state that no "
            "source material was retrieved, and do not cite anything."
        )

    lines = [
        "RETRIEVED EVIDENCE (verbatim from the user's indexed corpus — cite by "
        "number, e.g. [2], and never cite anything not listed here):"
    ]
    for i, passage in enumerate(passages, start=1):
        citation = getattr(passage, "citation", None)
        label = citation() if callable(citation) else str(passage)
        excerpt = getattr(passage, "excerpt", "")
        score = getattr(passage, "score", 0.0)
        lines.append(f"\n[{i}] {label} (relevance {score:.2f})\n{excerpt}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_messages(
    message: str,
    educational_mode: str = "Resident",
    specialty: str = "General Surgery",
    video_context: Optional[dict] = None,
    timeline: Optional[list[dict]] = None,
    passages: Optional[list] = None,
    history: Optional[list[dict]] = None,
    voice_mode: bool = False,
    ground_truth: Optional[dict] = None,
    language: str = "",
) -> list[ChatTurn]:
    """Assemble the full turn list sent to the language model."""
    from ai.knowledge import knowledge_graph

    phase = (video_context or {}).get("detected_phase")
    tools = (video_context or {}).get("tools") or []

    # Recorded annotations beat model output for grounding the specialty
    # context: if the dataset says the task is suturing, that is what the
    # knowledge graph should be primed with.
    if ground_truth and ground_truth.get("has_ground_truth"):
        phase = ground_truth.get("task") or phase
        tools = ground_truth.get("tools") or tools

    system_parts = [
        CORE_ROLE,
        audience_block(educational_mode),
        "SPECIALTY CONTEXT (from the platform's knowledge graph):\n"
        + knowledge_graph.context_for(specialty, phase, tools),
        groundtruth_block(ground_truth),
        observation_block(video_context, timeline),
        evidence_block(passages or []),
        f"Always close with this exact line on its own:\n> {settings.disclaimer}",
    ]

    # Placed after the audience block so it overrides any language implied
    # there, and before the disclaimer so the closing line is not the last
    # instruction the model reads about language.
    from ai.voice.languages import reply_instruction  # noqa: PLC0415

    instruction = reply_instruction(language)
    if instruction:
        system_parts.insert(2, instruction)

    if voice_mode:
        system_parts.insert(
            1,
            "DELIVERY: this answer will be spoken aloud. Write for the ear — "
            "plain sentences, no markdown, no bullet lists, no bracketed "
            "citation numbers, no URLs. Keep it under about 120 words and lead "
            "with the answer. Omit the closing disclaimer line; the interface "
            "displays it separately.",
        )
        system_parts[-1] = (
            "Do not append a disclaimer line in voice mode — the interface "
            "shows it on screen."
        )

    turns = [ChatTurn(role="system", content="\n\n---\n\n".join(system_parts))]

    # Recent history, trimmed. Long transcripts crowd out the evidence block,
    # which matters more to answer quality than distant turns.
    for turn in (history or [])[-6:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            turns.append(ChatTurn(role=role, content=content[:2000]))

    turns.append(ChatTurn(role="user", content=message))
    return turns


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
