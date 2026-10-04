"""
Tests for the detection path: COCO parsing, splits, and mAP.

These cover the failure modes that are silent — the ones where training still
converges and the reported number is simply wrong.
"""

from __future__ import annotations

import json

import pytest

from ai.datasets.coco import (
    DETECTION_CLASSES,
    Box,
    load_coco,
    split_clips,
)
from ai.training.detection_metrics import DetBox, average_precision_for_class, evaluate_detections, iou


# ---------------------------------------------------------------------------
# COCO parsing
# ---------------------------------------------------------------------------


def _write_coco(tmp_path, categories, annotations, images=None):
    payload = {
        "images": images
        or [{"id": 0, "width": 640, "height": 512, "file_name": "0000000000.jpg"}],
        "annotations": annotations,
        "categories": categories,
    }
    path = tmp_path / "clip_coco.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_categories_map_by_name_not_position(tmp_path):
    """
    A reordered category list must not relabel boxes.

    Mapping by position would turn this needle driver into a grasping
    retractor with no error raised anywhere — the model would train happily on
    systematically wrong labels.
    """
    categories = [
        {"id": 0, "name": "needle driver"},
        {"id": 1, "name": "grasping retractor"},
    ]
    path = _write_coco(
        tmp_path, categories, [{"category_id": 0, "image_id": 0, "bbox": [10, 10, 20, 20], "id": 0}]
    )
    clip = load_coco(path, "x", tmp_path / "x.mp4")
    assert clip.frames[0].boxes[0].class_name == "needle_driver"


def test_singular_and_plural_scissors_collapse_to_one_class(tmp_path):
    """
    cat1 spells it 'monopolar curved scissor'; surgvu24 spells it plural.

    If these stay distinct, presence labels and boxes silently refer to
    different classes and cross-corpus evaluation is nonsense.
    """
    path = _write_coco(
        tmp_path,
        [{"id": 0, "name": "monopolar curved scissor"}],
        [{"category_id": 0, "image_id": 0, "bbox": [1, 1, 5, 5], "id": 0}],
    )
    clip = load_coco(path, "x", tmp_path / "x.mp4")
    assert clip.frames[0].boxes[0].class_name == "monopolar_curved_scissors"
    assert "monopolar_curved_scissors" in DETECTION_CLASSES


def test_unknown_category_is_dropped_not_misassigned(tmp_path):
    path = _write_coco(
        tmp_path,
        [{"id": 7, "name": "sonic screwdriver"}],
        [{"category_id": 7, "image_id": 0, "bbox": [1, 1, 5, 5], "id": 0}],
    )
    clip = load_coco(path, "x", tmp_path / "x.mp4")
    assert clip.frames[0].boxes == []


def test_degenerate_boxes_are_dropped(tmp_path):
    path = _write_coco(
        tmp_path,
        [{"id": 0, "name": "needle driver"}],
        [
            {"category_id": 0, "image_id": 0, "bbox": [1, 1, 0, 10], "id": 0},
            {"category_id": 0, "image_id": 0, "bbox": [1, 1, 10, 10], "id": 1},
        ],
    )
    clip = load_coco(path, "x", tmp_path / "x.mp4")
    assert len(clip.frames[0].boxes) == 1


def test_yolo_conversion_is_normalised_and_centred():
    box = Box(class_id=0, x=100.0, y=50.0, w=200.0, h=100.0)
    cx, cy, w, h = box.to_yolo(width=400, height=200)
    assert cx == pytest.approx(0.5)   # (100 + 100) / 400
    assert cy == pytest.approx(0.5)   # (50 + 50) / 200
    assert w == pytest.approx(0.5)
    assert h == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def test_split_is_by_clip_and_disjoint():
    train, val = split_clips(["1", "2", "3", "4", "5", "6", "7"])
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == {"1", "2", "3", "4", "5", "6", "7"}
    assert train and val


def test_split_is_deterministic():
    assert split_clips(list("1234567")) == split_clips(list("7654321"))


def test_split_never_empties_train():
    train, val = split_clips(["a", "b"], val_fraction=0.99)
    assert train and val


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_iou_basics():
    a = DetBox("i", 0, 0, 0, 10, 10)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, DetBox("i", 0, 20, 20, 30, 30)) == 0.0
    # Half-overlap: intersection 50, union 150.
    assert iou(a, DetBox("i", 0, 5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_perfect_predictions_score_one():
    gt = [DetBox("a", 0, 0, 0, 10, 10), DetBox("b", 0, 5, 5, 15, 15)]
    preds = [DetBox(b.image_id, 0, b.x1, b.y1, b.x2, b.y2, score=0.9) for b in gt]
    report = evaluate_detections(preds, gt, DETECTION_CLASSES)
    assert report["mAP50"] == pytest.approx(1.0)


def test_each_ground_truth_box_is_matched_at_most_once():
    """
    Emitting twenty boxes on one instrument must not buy twenty true positives.

    Without one-to-one matching, a localiser that floods every frame reports
    near-100% recall — precisely the shortcut a weakly-supervised method is
    prone to, since a broad heatmap yields many overlapping candidates.
    """
    gt = [DetBox("a", 0, 0, 0, 10, 10)]
    flooded = [DetBox("a", 0, 0, 0, 10, 10, score=1.0 - i * 0.01) for i in range(20)]
    ap, tp, n_gt = average_precision_for_class(flooded, gt, 0.5)
    assert (tp, n_gt) == (1, 1)
    # AP itself stays 1.0 here, and that is correct: AP is rank-based, and
    # these 19 false positives all rank *below* the true positive. See the
    # next test for the case that does move the number.
    assert ap == pytest.approx(1.0)


def test_false_positives_ranked_above_the_match_do_reduce_ap():
    """
    The failure that AP is actually sensitive to: confident wrong boxes.

    A localiser whose spurious candidates outrank its correct one is penalised,
    which is the property that makes AP worth reporting at all.
    """
    gt = [DetBox("a", 0, 0, 0, 10, 10)]
    preds = [
        DetBox("a", 0, 500, 500, 510, 510, score=0.99),  # confident, nowhere near
        DetBox("a", 0, 400, 400, 410, 410, score=0.98),
        DetBox("a", 0, 0, 0, 10, 10, score=0.10),        # the real one, ranked last
    ]
    ap, tp, _ = average_precision_for_class(preds, gt, 0.5)
    assert tp == 1
    assert ap == pytest.approx(1 / 3, abs=1e-6)


def test_classes_without_ground_truth_are_excluded_not_zeroed():
    """
    Six of fourteen cat1 classes never occur. Averaging six zeroes in would
    report a mAP under half the real value.
    """
    gt = [DetBox("a", 11, 0, 0, 10, 10)]
    preds = [DetBox("a", 11, 0, 0, 10, 10, score=0.9)]
    report = evaluate_detections(preds, gt, DETECTION_CLASSES)
    assert report["mAP50"] == pytest.approx(1.0)
    assert report["classes_evaluated"] == 1
    assert len(report["classes_skipped_no_ground_truth"]) == len(DETECTION_CLASSES) - 1


def test_wrong_class_prediction_does_not_match():
    gt = [DetBox("a", 11, 0, 0, 10, 10)]
    preds = [DetBox("a", 2, 0, 0, 10, 10, score=0.9)]
    report = evaluate_detections(preds, gt, DETECTION_CLASSES)
    assert report["mAP50"] == 0.0


def test_loose_boxes_pass_at_50_but_fail_at_strict_iou():
    """
    The characteristic signature of a heatmap-derived box: roughly right,
    loosely placed. mAP50 stays high while mAP50_95 collapses, which is why
    both are always reported together.
    """
    gt = [DetBox("a", 0, 0, 0, 100, 100)]
    loose = [DetBox("a", 0, -12, -12, 88, 88, score=0.9)]  # IoU ≈ 0.6
    report = evaluate_detections(loose, gt, DETECTION_CLASSES)
    assert report["mAP50"] == pytest.approx(1.0)
    assert report["mAP50_95"] < 0.35


def test_empty_predictions_score_zero_not_crash():
    gt = [DetBox("a", 0, 0, 0, 10, 10)]
    report = evaluate_detections([], gt, DETECTION_CLASSES)
    assert report["mAP50"] == 0.0
    assert report["total_predictions"] == 0
