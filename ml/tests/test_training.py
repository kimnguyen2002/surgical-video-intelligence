"""
SurgVU pipeline: label parsing, metrics, and model construction.

The label-parsing tests matter most. SurgVU labels are intervals keyed by
(case, part, time) that can span video parts, and getting that resolution wrong
silently mislabels frames — the model trains, the loss falls, and the result is
meaningless.
"""

from __future__ import annotations

import math

import pytest

from ai.training.labels import (
    IntervalIndex,
    normalise_label,
    parse_case_part,
    parse_time,
    resolve_intervals,
)
from ai.training.metrics import (
    average_precision,
    mean_average_precision,
    multiclass_report,
    multilabel_report,
    precision_recall_f1,
)


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Monopolar Curved Scissors", "monopolar_curved_scissors"),
        ("needle driver", "needle_driver"),
        ("rectal artery/vein", "rectal_artery_vein"),
        ("permanent cautery hook/spatula", "permanent_cautery_hook_spatula"),
        ("cautery hook", "permanent_cautery_hook_spatula"),
        ("Tip-Up Fenestrated Grasper", "tip_up_fenestrated_grasper"),
        ("General Skills Application", "skills_application"),
        ("", "other"),
    ],
)
def test_label_normalisation(raw, expected):
    assert normalise_label(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("case_042_video_part_003", ("042", 3)),
        ("case_7_part_1", ("7", 1)),
        ("CASE_015_VIDEO_PART_012", ("015", 12)),
    ],
)
def test_case_part_parsing(raw, expected):
    assert parse_case_part(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("125.5", 125.5), ("00:02:05", 125.0), ("2:05", 125.0), ("", 0.0), ("junk", 0.0)],
)
def test_time_parsing(raw, expected):
    assert parse_time(raw) == pytest.approx(expected)


def test_interval_spanning_multiple_parts():
    """
    A tool installed in part 1 and removed in part 3 must cover all of part 2.
    Dropping the middle part would silently label those frames "no tool".
    """
    rows = [
        {
            "case": "001", "start_part": 1, "start_time": 100.0,
            "end_case": "001", "end_part": 3, "end_time": 50.0,
            "arm": "1", "label": "needle_driver",
        }
    ]
    durations = {("001", 1): 300.0, ("001", 2): 300.0, ("001", 3): 300.0}

    intervals = resolve_intervals(rows, durations)
    by_part = {i.part: i for i in intervals}

    assert set(by_part) == {1, 2, 3}
    assert (by_part[1].start, by_part[1].end) == (100.0, 300.0)
    assert (by_part[2].start, by_part[2].end) == (0.0, 300.0)   # full middle part
    assert (by_part[3].start, by_part[3].end) == (0.0, 50.0)


def test_parts_without_frames_are_skipped_not_guessed():
    rows = [
        {
            "case": "001", "start_part": 1, "start_time": 0.0,
            "end_case": "001", "end_part": 5, "end_time": 10.0,
            "arm": "1", "label": "stapler",
        }
    ]
    intervals = resolve_intervals(rows, {("001", 1): 100.0})
    assert [i.part for i in intervals] == [1]


def test_interval_index_returns_all_concurrent_tools():
    """Several arms carry tools at once — presence is genuinely multi-label."""
    rows = [
        {"case": "1", "start_part": 0, "start_time": 0.0, "end_case": "1",
         "end_part": 0, "end_time": 100.0, "arm": "1", "label": "needle_driver"},
        {"case": "1", "start_part": 0, "start_time": 50.0, "end_case": "1",
         "end_part": 0, "end_time": 150.0, "arm": "2", "label": "bipolar_forceps"},
    ]
    index = IntervalIndex(resolve_intervals(rows, {("1", 0): 200.0}))

    assert index.labels_at("1", 0, 25) == {"needle_driver"}
    assert index.labels_at("1", 0, 75) == {"needle_driver", "bipolar_forceps"}
    assert index.labels_at("1", 0, 120) == {"bipolar_forceps"}
    assert index.labels_at("1", 0, 180) == frozenset()


def test_label_counts_measure_seconds_per_class():
    rows = [
        {"case": "1", "start_part": 0, "start_time": 0.0, "end_case": "1",
         "end_part": 0, "end_time": 60.0, "arm": "1", "label": "needle_driver"},
    ]
    index = IntervalIndex(resolve_intervals(rows, {("1", 0): 100.0}))
    assert index.label_counts()["needle_driver"] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_average_precision_is_one_for_perfect_ranking():
    assert average_precision([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_average_precision_penalises_bad_ranking():
    good = average_precision([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    bad = average_precision([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1])
    assert bad < good


def test_average_precision_undefined_without_positives():
    assert math.isnan(average_precision([0.5, 0.4], [0, 0]))


def test_map_skips_classes_with_no_positives():
    scores = [[0.9, 0.1], [0.8, 0.2]]
    targets = [[1, 0], [1, 0]]
    value, per_class = mean_average_precision(scores, targets)
    assert value == pytest.approx(1.0)
    assert math.isnan(per_class[1])


def test_map_is_not_fooled_by_the_all_negative_shortcut():
    """
    The reason mAP is the headline metric: with rare classes, predicting
    'absent' everywhere scores high accuracy and near-zero mAP.
    """
    # One positive in twenty samples; the model always predicts absent.
    scores = [[0.01] for _ in range(20)]
    targets = [[1]] + [[0] for _ in range(19)]

    report = multilabel_report(scores, targets, ["rare_tool"])
    stats = report["per_class"]["rare_tool"]

    assert stats["recall"] == 0.0          # it found nothing
    assert stats["f1"] == 0.0
    assert report["exact_match"] < 1.0     # and it is not perfect


def test_precision_recall_f1_counts():
    stats = precision_recall_f1([0.9, 0.8, 0.2, 0.1], [1, 0, 1, 0], threshold=0.5)
    assert (stats["tp"], stats["fp"], stats["fn"], stats["tn"]) == (1, 1, 1, 1)
    assert stats["precision"] == pytest.approx(0.5)
    assert stats["recall"] == pytest.approx(0.5)


def test_balanced_accuracy_exposes_majority_class_bias():
    """
    'other' dominates SurgVU tasks. Plain accuracy rewards always predicting it;
    balanced accuracy does not, which is why it selects the checkpoint.
    """
    targets = [0] * 90 + [1] * 10
    predictions = [0] * 100  # always the majority class

    report = multiclass_report(predictions, targets, ["other", "suturing"])
    assert report["accuracy"] == pytest.approx(0.9)
    assert report["balanced_accuracy"] == pytest.approx(0.5)


def test_confusion_matrix_orientation():
    report = multiclass_report([0, 1, 1], [0, 1, 0], ["a", "b"])
    # rows = truth, cols = prediction
    assert report["confusion_matrix"] == [[1, 1], [0, 1]]


# ---------------------------------------------------------------------------
# Models (torch required)
# ---------------------------------------------------------------------------
@pytest.mark.requires_torch
def test_tool_model_shape_and_multilabel_activation():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")

    from ai.training.models import SurgVUToolModel

    model = SurgVUToolModel(model_name="convnext_tiny", num_classes=12, pretrained=False)
    model.eval()

    with torch.no_grad():
        logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 12)

    # Sigmoid, not softmax: several instruments are present at once, so the
    # per-class probabilities must not be forced to sum to 1.
    probabilities = torch.sigmoid(logits)
    assert not torch.allclose(probabilities.sum(dim=1), torch.ones(2))


@pytest.mark.requires_torch
def test_step_model_consumes_a_clip():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")

    from ai.training.models import SurgVUStepModel

    model = SurgVUStepModel(
        model_name="cnn_gru", num_classes=8, frame_encoder="convnext_tiny",
        pretrained=False, clip_length=4,
    )
    model.eval()

    with torch.no_grad():
        logits, attention = model(torch.randn(1, 4, 3, 224, 224), return_attention=True)

    assert logits.shape == (1, 8)
    assert attention.shape == (1, 4)
    # Attention weights over time must be a distribution.
    assert attention.sum().item() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.requires_torch
def test_gradcam_produces_a_spatial_map():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")

    from ai.explainability import GradCAM
    from ai.training.models import SurgVUToolModel

    model = SurgVUToolModel(model_name="convnext_tiny", num_classes=12, pretrained=False)
    model.eval()

    layer = model.gradcam_layer()
    assert layer is not None

    with GradCAM(model, layer) as cam:
        heatmap = cam.generate(torch.randn(1, 3, 224, 224), class_idx=3)

    assert heatmap is not None
    assert heatmap.shape == (224, 224)
    assert 0.0 <= float(heatmap.min()) and float(heatmap.max()) <= 1.0


@pytest.mark.requires_torch
def test_checkpoint_roundtrip_restores_architecture(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("timm")

    from ai.training.models import SurgVUToolModel, load_checkpoint, save_checkpoint

    model = SurgVUToolModel(model_name="convnext_tiny", num_classes=12, pretrained=False)
    path = tmp_path / "tool.pt"
    save_checkpoint(path, model, {"task": "tool_detection", "classes": ["a"] * 12})

    # Loading must rebuild the model from the file alone — a bare state_dict
    # would require the caller to already know the architecture.
    restored, state = load_checkpoint(path, task="tool", device="cpu")
    assert restored.model_name == "convnext_tiny"
    assert state["task"] == "tool_detection"
