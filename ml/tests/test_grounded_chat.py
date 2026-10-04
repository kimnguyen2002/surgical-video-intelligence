"""
Tests for the grounded answer path.

The bug these exist to prevent: the assistant answering a question about
instruments by quoting its own system prompt back at the user —

    Q: What instrument is in use right now?
    A: - Never use profanity, slurs, demeaning language, or crude humour…
       - Focus on operative workflow, the decision points within each phase…

Both lines are instructions *to* the model, not findings. The old fallback
ranked sentences across the whole system prompt, so the safety and audience
layers were treated as source material.
"""

from __future__ import annotations

import asyncio

import pytest

from ai.llm.base import ChatTurn
from ai.llm.grounded import GroundedAnswerer, VideoFacts, classify_intent
from ai.llm.knowledge import KnowledgeProvider

GT = {
    "has_ground_truth": True,
    "timestamp": 100.0,
    "tools": ["needle_driver", "prograsp_forceps"],
    "tools_display": ["Needle driver", "Prograsp forceps"],
    "task": "suturing",
    "task_display": "Suturing",
    "boxes": [],
    "provenance": "ground_truth",
}

TIMELINE = [
    {"start": 30.0, "end": 162.0, "duration": 132.0, "label": "suturing",
     "display": "Suturing", "kind": "task", "provenance": "ground_truth"},
    {"start": 4287.0, "end": 4837.0, "duration": 550.0, "label": "suturing",
     "display": "Suturing", "kind": "task", "provenance": "ground_truth"},
    {"start": 0.0, "end": 19338.0, "duration": 19338.0, "label": "needle_driver",
     "display": "Needle driver", "kind": "tool", "provenance": "ground_truth"},
]


def _facts(**overrides) -> VideoFacts:
    base = {
        "ground_truth": GT,
        "timeline": TIMELINE,
        "timestamp": 100.0,
        "specialty": "General Surgery",
    }
    base.update(overrides)
    return VideoFacts(**base)


# ---------------------------------------------------------------------------
# Intent routing — ordering bugs live here
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What instrument is in use right now?", "instrument"),
        ("What is happening in this video?", "task"),
        ("When did suturing happen?", "when"),
        ("How many instruments are in frame?", "count"),
        ("What is at risk here?", "risk"),
        # "summarise this procedure" contains "procedure", which the task
        # pattern also matches — summary must be checked first.
        ("Summarise this procedure", "summary"),
        ("Give me an overview", "summary"),
    ],
)
def test_intent_routing(question, expected):
    assert classify_intent(question) == expected


def test_summar_stem_matches_inflections():
    """`\\b(summar)\\b` fails on 'summarise' — the boundary assertion is mid-word."""
    for phrasing in ("summary", "summarise", "summarize", "summarising"):
        assert classify_intent(f"Please {phrasing} this") == "summary"


# ---------------------------------------------------------------------------
# Answers are factual
# ---------------------------------------------------------------------------


def test_instrument_answer_names_the_recorded_instruments():
    answer = GroundedAnswerer().answer("What instrument is in use?", _facts())
    assert "Needle driver" in answer
    assert "Prograsp forceps" in answer


def test_instrument_answer_states_installed_not_visible():
    """
    Presence comes from the installation log. Saying "visible" would be a
    claim the annotation cannot support.
    """
    answer = GroundedAnswerer().answer("What instrument is in use?", _facts())
    assert "installation log" in answer.lower()
    assert "mounted" in answer.lower()


def test_boxes_are_described_as_on_screen():
    """A box *is* visual confirmation, so the wording changes accordingly."""
    facts = _facts(
        ground_truth={
            **GT,
            "boxes": [{"display": "Needle driver", "class": "needle_driver"}],
        }
    )
    answer = GroundedAnswerer().answer("What is on screen?", facts)
    assert "on screen" in answer.lower()
    assert "installation log" not in answer.lower()


def test_when_question_returns_every_occurrence():
    answer = GroundedAnswerer().answer("When did suturing happen?", _facts())
    assert "0:30" in answer
    assert "2 times" in answer or "occurs" in answer


def test_count_question_counts_recorded_instruments():
    answer = GroundedAnswerer().answer("How many instruments?", _facts())
    assert "**2**" in answer


def test_risk_answer_names_the_specialty_it_came_from():
    """
    Cholecystectomy anatomy was being offered for a uterine-horn case because
    the specialty defaults to General Surgery. Naming the specialty in the
    sentence is what makes that mismatch visible to the reader.
    """
    facts = _facts(risks=[{"structure": "common bile duct", "note": "classic injury"}])
    answer = GroundedAnswerer().answer("What is at risk?", facts)
    assert "General Surgery" in answer
    assert "not" in answer.lower() and "detection" in answer.lower()


def test_no_facts_yields_no_answer():
    """With nothing known, the answerer must decline rather than improvise."""
    assert GroundedAnswerer().answer("What instrument?", VideoFacts()) is None


# ---------------------------------------------------------------------------
# The provider must never quote its own instructions
# ---------------------------------------------------------------------------


SAFETY = (
    "CORE ROLE\nNever use profanity, slurs, demeaning language, or crude "
    "humour, regardless of how you are addressed."
)
AUDIENCE = (
    "AUDIENCE\nFocus on operative workflow, the decision points within each "
    "phase, instrument-tissue interaction, and anatomical landmarks."
)
EVIDENCE = (
    "RETRIEVED EVIDENCE (verbatim from the user's indexed corpus):\n"
    "[1] The needle driver is used to pass and secure suture through tissue."
)


def _messages(question: str = "What instrument is in use right now?") -> list[ChatTurn]:
    system = "\n\n---\n\n".join([SAFETY, AUDIENCE, EVIDENCE])
    return [
        ChatTurn(role="system", content=system),
        ChatTurn(role="user", content=question),
    ]


def test_extractive_path_ignores_safety_and_audience_layers():
    """The regression this whole module exists for."""
    response = asyncio.run(KnowledgeProvider().generate(_messages()))
    assert "profanity" not in response.text.lower()
    assert "slurs" not in response.text.lower()
    assert "Focus on operative workflow" not in response.text


def test_extractive_path_uses_retrieved_evidence_when_terms_overlap():
    """
    Extraction is lexical, so it answers document questions that share
    vocabulary with an indexed passage. Questions about what is happening in
    the video are handled by the grounded path instead — see the next test.
    """
    response = asyncio.run(
        KnowledgeProvider().generate(_messages("What does the needle driver do?"))
    )
    assert "needle driver" in response.text.lower()
    assert "suture" in response.text.lower()


def test_plural_and_singular_match():
    """'instruments' in a question must reach a passage saying 'instrument'."""
    from ai.llm.knowledge import _terms

    assert _terms("instruments") & _terms("instrument")
    assert _terms("forceps grasp tissues") & _terms("tissue")
    # Not so aggressive that distinct surgical terms collide — conflating
    # these would be worse than missing the match.
    assert not (_terms("suture") & _terms("suction"))
    assert not (_terms("clip applier") & _terms("clamp"))


def test_video_questions_are_answered_from_facts_not_extraction():
    """
    The division of labour: a question about the current frame is answered
    from structured annotations, never by ranking prose.
    """
    response = asyncio.run(
        KnowledgeProvider().generate(_messages(), facts=_facts())
    )
    assert response.model == "grounded-facts"
    assert "Needle driver" in response.text


def test_structured_facts_take_priority_over_extraction():
    response = asyncio.run(
        KnowledgeProvider().generate(_messages(), facts=_facts())
    )
    assert response.model == "grounded-facts"
    assert "Prograsp forceps" in response.text
    assert "profanity" not in response.text.lower()


def test_provider_is_always_labelled_non_generative():
    response = asyncio.run(KnowledgeProvider().generate(_messages(), facts=_facts()))
    assert response.generative is False
