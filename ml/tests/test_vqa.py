"""Tests for cat2 VQA scoring."""

from __future__ import annotations

import pytest

from ai.datasets.vqa import (
    aggregate,
    normalise_answer,
    polarity,
    score_answer,
    token_f1,
)


def test_normalisation_strips_punctuation_and_articles():
    assert normalise_answer("No, a large needle driver is not listed.") == (
        "no large needle driver is not listed"
    )


def test_polarity_reads_the_leading_stance():
    """
    'No, a large needle driver is not listed' contains 'listed', a positive
    marker. Counting markers across the whole string would misread it; a human
    reads the stance off the first word.
    """
    assert polarity("No, a large needle driver is not listed.") is False
    assert polarity("No large needle driver is included.") is False
    assert polarity("Yes, forceps are clearly in use.") is True
    assert polarity("There is a needle driver present.") is True


def test_polarity_none_when_no_stance_taken():
    assert polarity("The video shows a surgical field.") is None
    assert polarity("") is None


def test_token_f1_behaviour():
    assert token_f1("no", "no") == pytest.approx(1.0)
    assert token_f1("yes", "no") == 0.0
    assert 0 < token_f1("no needle driver listed", "no large needle driver is listed") < 1


def test_verbose_but_correct_answer_scores_on_polarity():
    """
    The realistic case: the assistant answers correctly but at length. Exact
    match fails; polarity is what actually reflects correctness here.
    """
    score = score_answer(
        "case123",
        "Is a large needle driver among the listed tools?",
        "Based on the dataset annotations for this clip, no — a large needle "
        "driver does not appear among the instruments recorded.",
        ["No", "No, a large needle driver is not listed."],
    )
    assert score.exact is False
    assert score.polarity_match is True


def test_confidently_wrong_answer_is_caught():
    score = score_answer(
        "case123",
        "Is a large needle driver among the listed tools?",
        "Yes, a large needle driver is clearly among the listed tools.",
        ["No", "No, a large needle driver is not listed."],
    )
    assert score.polarity_match is False


def test_exact_match_recognised():
    score = score_answer("c", "q?", "No.", ["No", "No, it is not."])
    assert score.exact is True
    assert score.best_f1 == pytest.approx(1.0)


def test_aggregate_reports_abstentions_separately():
    """
    A system that hedges on everything makes no polarity errors. If abstentions
    were not counted, that would read as perfect accuracy.
    """
    scores = [
        score_answer("a", "q", "No.", ["No"]),
        score_answer("b", "q", "It is difficult to say.", ["Yes"]),
    ]
    summary = aggregate(scores)
    assert summary["cases"] == 2
    assert summary["polarity_scored"] == 1
    assert summary["polarity_abstained"] == 1
    assert summary["polarity_accuracy"] == pytest.approx(1.0)


def test_aggregate_empty():
    assert aggregate([])["cases"] == 0
