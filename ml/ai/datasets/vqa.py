"""
cat2 visual question answering.

Each cat2 case is a short clip, one free-text question, and a list of accepted
answer phrasings — "No", "No, a large needle driver is not listed.", and so on.
The task is open-ended, so scoring is not string equality.

This is an **evaluation harness, not a training set**. Seven cases cannot train
anything; what they can do is measure whether the assistant, given the video
evidence this platform already extracts, answers correctly. That makes cat2 a
regression test on the end-to-end grounding chain — detector and annotations →
prompt → language model → answer — which is exactly the chain most likely to
break silently when any one link changes.

Scoring uses two measures deliberately:

``exact``   the normalised answer matches an accepted phrasing outright.
``polarity`` the yes/no stance agrees.

Most cat2 questions are polarity questions ("Is a large needle driver among the
listed tools?"), and for those the stance *is* the answer — a reply that is
verbose but says "no" is correct, while one that says "yes" politely is not.
Reporting only exact match would score a correct system near zero; reporting
only polarity would call a hedging non-answer correct. Both are printed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Optional

logger = logging.getLogger("charlie.datasets.vqa")

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")

_NEGATIVE_MARKERS = (
    "no,", "no.", "no ", "not ", "isn't", "is not", "aren't", "are not",
    "doesn't", "does not", "don't", "do not", "none", "neither", "never",
    "absent", "no such", "cannot", "can't", "unable",
)
#: Kept specific on purpose. Generic copulas like "it is" and "they are" match
#: hedges ("it is difficult to say") as readily as affirmations, which turns
#: every non-answer into a confident yes.
_POSITIVE_MARKERS = (
    "yes,", "yes.", "yes ", "correct", "indeed", "affirmative",
    "there is a", "there are", "is present", "are present", "is visible",
    "are visible", "is included", "are included", "is listed", "are listed",
    "does appear", "do appear", "can be seen",
)

#: An answer that hedges has taken no stance, however many other markers it
#: happens to contain. Scoring it as agreement would let a system that never
#: commits to anything post perfect polarity accuracy.
_HEDGE_MARKERS = (
    "difficult to say", "hard to say", "unclear", "uncertain", "cannot tell",
    "can't tell", "not sure", "unsure", "insufficient", "no way to tell",
    "impossible to say", "i don't know", "i do not know",
)


def normalise_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    text = (text or "").strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
    return " ".join(tokens)


def polarity(text: str) -> Optional[bool]:
    """
    The yes/no stance of an answer, or ``None`` when it takes none.

    Leading markers win. "No, a large needle driver is not listed" contains
    "listed", a positive marker — counting markers anywhere would score it as
    ambiguous or positive, when a human reads the stance off the first word.
    """
    lowered = f" {(text or '').strip().lower()} "

    # A hedge overrides everything else — it is the absence of a stance.
    if any(marker in lowered for marker in _HEDGE_MARKERS):
        return None

    head = lowered[:24]

    for marker in ("no,", "no.", " no ", "nope"):
        if marker in head:
            return False
    for marker in ("yes,", "yes.", " yes ", "yep"):
        if marker in head:
            return True

    negative = sum(1 for m in _NEGATIVE_MARKERS if m in lowered)
    positive = sum(1 for m in _POSITIVE_MARKERS if m in lowered)
    if negative > positive:
        return False
    if positive > negative:
        return True
    return None


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 — the standard open-ended QA similarity."""
    p_tokens = normalise_answer(prediction).split()
    r_tokens = normalise_answer(reference).split()
    if not p_tokens or not r_tokens:
        return float(p_tokens == r_tokens)

    common: dict[str, int] = {}
    for token in p_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in r_tokens:
        if common.get(token, 0) > 0:
            common[token] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_tokens)
    recall = overlap / len(r_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass
class VQAScore:
    case: str
    question: str
    prediction: str
    accepted: tuple[str, ...]
    exact: bool
    best_f1: float
    polarity_match: Optional[bool]

    def to_dict(self) -> dict:
        return {
            "case": self.case,
            "question": self.question,
            "prediction": self.prediction,
            "accepted": list(self.accepted),
            "exact": self.exact,
            "best_f1": round(self.best_f1, 4),
            "polarity_match": self.polarity_match,
        }


def score_answer(
    case: str, question: str, prediction: str, accepted: Iterable[str]
) -> VQAScore:
    """Score one predicted answer against every accepted phrasing."""
    accepted = tuple(accepted)
    normalised = normalise_answer(prediction)

    exact = any(normalise_answer(a) == normalised for a in accepted)
    best_f1 = max((token_f1(prediction, a) for a in accepted), default=0.0)

    predicted_polarity = polarity(prediction)
    reference_polarities = {polarity(a) for a in accepted} - {None}
    if predicted_polarity is None or not reference_polarities:
        polarity_match = None
    else:
        polarity_match = predicted_polarity in reference_polarities

    return VQAScore(
        case=case,
        question=question,
        prediction=prediction,
        accepted=accepted,
        exact=exact,
        best_f1=best_f1,
        polarity_match=polarity_match,
    )


def aggregate(scores: list[VQAScore]) -> dict:
    """Summary over a set of scored answers."""
    if not scores:
        return {
            "cases": 0,
            "exact_match": 0.0,
            "mean_f1": 0.0,
            "polarity_accuracy": None,
            "polarity_scored": 0,
        }

    judged = [s for s in scores if s.polarity_match is not None]
    return {
        "cases": len(scores),
        "exact_match": round(sum(s.exact for s in scores) / len(scores), 4),
        "mean_f1": round(sum(s.best_f1 for s in scores) / len(scores), 4),
        "polarity_accuracy": (
            round(sum(bool(s.polarity_match) for s in judged) / len(judged), 4)
            if judged
            else None
        ),
        # How many answers took a stance at all. A system that hedges on
        # everything scores no polarity errors, which would look like success.
        "polarity_scored": len(judged),
        "polarity_abstained": len(scores) - len(judged),
    }
