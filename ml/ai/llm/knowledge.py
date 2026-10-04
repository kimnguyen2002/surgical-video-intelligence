"""
Knowledge provider — the last-resort answer path.

When no language model is reachable (no Ollama, no API key), the assistant
still has to be useful. This provider does *extractive* composition rather
than generation: it ranks the sentences already present in the retrieved
context — RAG passages, timeline events, specialty ontology facts — against
the question, and assembles the best-matching ones into an answer.

That means answers vary with the corpus and the video rather than replaying a
fixed script, and every sentence is traceable to indexed material. It is
labelled ``generative=False`` throughout the stack so the UI can say plainly
that this is retrieved material, not a generated explanation — the one thing
worse than a thin answer is a fluent one with nothing behind it.
"""

from __future__ import annotations

import logging
import re
import time

from .base import ChatTurn, LLMProvider, LLMResponse

logger = logging.getLogger("charlie.llm.knowledge")

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "for",
    "on", "with", "as", "by", "at", "from", "that", "this", "it", "be", "what",
    "how", "why", "when", "where", "which", "who", "do", "does", "did", "can",
    "you", "i", "me", "my", "we", "please", "tell", "show", "explain",
}


def _stem(word: str) -> str:
    """
    Strip a plural suffix.

    Deliberately minimal: it exists to stop "instruments" and "instrument"
    being treated as unrelated terms, which is the most common reason a lexical
    match misses a passage that plainly answers the question.

    It does **not** attempt verb forms. Undoubling a final consonant would be
    needed for "clipping" → "clip", and that same rule turns "press" into
    "pres" while "pressing" also becomes "pres" — matching by accident in one
    direction and failing in the other. Aggressive stemming also collides
    distinct surgical terms, which is worse than missing a match. Questions
    about the video are answered from structured facts by
    :mod:`ai.llm.grounded`, not by this ranker, so the cost of the limitation
    is small.
    """
    if len(word) < 4 or not word.endswith("s"):
        return word

    # "process", "status" — the trailing s is part of the word.
    if word.endswith(("ss", "us", "is")):
        return word

    # "arteries" -> "artery"
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    # "es" comes off only after a sibilant, where it is a real plural marker:
    # "boxes" -> "box", "arches" -> "arch". Applying it everywhere turns
    # "tissues" into "tissu" while "tissue" stays whole, so the two stop
    # matching — the exact failure this function is meant to prevent.
    if word.endswith("es") and word[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return word[:-2]

    return word[:-1]


def _terms(text: str) -> set[str]:
    return {
        _stem(w)
        for w in _WORD.findall(text.lower())
        if w not in _STOP and len(w) > 2
    }


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 25]


class KnowledgeProvider(LLMProvider):
    name = "knowledge"
    model = "extractive-composer"
    generative = False

    def is_available(self) -> bool:
        return True  # Always. That is the point of this provider.

    async def generate(self, messages: list[ChatTurn], **kwargs) -> LLMResponse:
        started = time.time()

        question = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )

        # 1. Structured facts first. For "what instrument is in use?" the
        #    dataset annotation *is* the answer; no amount of sentence ranking
        #    over prose improves on it.
        facts = kwargs.get("facts")
        if facts is not None:
            from .grounded import answerer  # noqa: PLC0415

            direct = answerer.answer(question, facts)
            if direct:
                return LLMResponse(
                    text=direct,
                    provider=self.name,
                    model="grounded-facts",
                    generative=False,
                    latency_ms=round((time.time() - started) * 1000, 1),
                )

        # 2. Otherwise extract from retrieved evidence only.
        answer = self._compose(question, self._evidence_only(messages))

        return LLMResponse(
            text=answer,
            provider=self.name,
            model=self.model,
            generative=False,
            latency_ms=round((time.time() - started) * 1000, 1),
        )

    @staticmethod
    def _evidence_only(messages: list[ChatTurn]) -> str:
        """
        The retrieved-evidence and ground-truth sections of the prompt — and
        nothing else.

        The system prompt is layered: safety constraints, audience calibration,
        specialty context, ground truth, observations, retrieved evidence.
        Only the last three are *findings*; the first two are instructions to
        the model. Mining all of them is what produced answers like "Never use
        profanity, slurs, demeaning language…" to a question about instruments.

        Sections are separated by the `\\n\\n---\\n\\n` delimiter that
        ``build_messages`` joins them with, so they can be split apart exactly
        rather than guessed at by keyword.
        """
        keep_markers = (
            "RECORDED GROUND TRUTH",
            "RETRIEVED EVIDENCE",
            "OBSERVED CONTEXT",
        )
        kept: list[str] = []
        for message in messages:
            if message.role != "system":
                continue
            for section in message.content.split("\n\n---\n\n"):
                if any(section.lstrip().startswith(m) for m in keep_markers):
                    kept.append(section)
        return "\n".join(kept)

    def _compose(self, question: str, context: str) -> str:
        query_terms = _terms(question)
        candidates = _sentences(context)

        scored: list[tuple[float, str]] = []
        for sentence in candidates:
            sentence_terms = _terms(sentence)
            if not sentence_terms:
                continue
            overlap = len(query_terms & sentence_terms)
            if not overlap:
                continue
            # Jaccard-ish score, mildly favouring information-dense sentences.
            score = overlap / (len(query_terms | sentence_terms) ** 0.5)
            scored.append((score, sentence))

        scored.sort(key=lambda pair: -pair[0])

        # De-duplicate near-identical sentences from overlapping chunks.
        selected: list[str] = []
        seen: list[set[str]] = []
        for _, sentence in scored:
            terms = _terms(sentence)
            if any(len(terms & prior) / max(len(terms), 1) > 0.7 for prior in seen):
                continue
            selected.append(sentence)
            seen.append(terms)
            if len(selected) >= 5:
                break

        if not selected:
            return (
                "I could not find anything in the indexed material that answers "
                "that.\n\n"
                "This instance has no language model connected, so I can only "
                "report what has actually been indexed — I will not improvise an "
                "answer. Two things would help:\n\n"
                "- **Upload reference material** (PDF, slides, notes) in the "
                "Knowledge panel so I have a corpus to draw on.\n"
                "- **Start a local model** with `ollama pull qwen2.5:7b && ollama serve` "
                "to enable generated explanations.\n\n"
                "You can also ask about the current video's timeline, detected "
                "phases, or instruments — those come from the analysis itself."
            )

        body = "\n\n".join(f"- {s}" for s in selected)
        return (
            "Here is what the indexed material says about that:\n\n"
            f"{body}\n\n"
            "_These passages are quoted from the indexed corpus and procedure "
            "timeline. No language model is connected to this instance, so "
            "nothing above is a generated explanation._"
        )
