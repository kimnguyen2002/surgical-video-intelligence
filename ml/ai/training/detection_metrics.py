"""
Detection metrics — IoU, per-class average precision, mAP.

Written out rather than pulled from ``pycocotools`` because both the
Ultralytics detector and the Grad-CAM localiser must be scored by *identical*
code for their numbers to be comparable. Two mAPs computed by different
implementations, with different interpolation and different handling of
unmatched classes, are not a comparison.

Conventions, all of which change the number if you get them wrong:

* Boxes are ``(x1, y1, x2, y2)`` in absolute pixels.
* A prediction matches a ground-truth box of the same class with IoU ≥ the
  threshold. Each ground-truth box may be matched **once**; further predictions
  on it are false positives, which is what stops a model from farming recall by
  emitting twenty boxes per instrument.
* Predictions are greedily matched in descending confidence within an image.
* AP is the area under the precision–recall curve by **continuous
  interpolation** (VOC 2010 / COCO style), not the 11-point sample.
* A class with no ground-truth boxes anywhere is excluded from mAP rather than
  scored 0. Six of the fourteen cat1 classes never occur; averaging in six
  zeroes would report a mAP under half the real value and make every
  comparison meaningless.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class DetBox:
    """One box, predicted or ground truth."""

    image_id: str
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def iou(a: DetBox, b: DetBox) -> float:
    """Intersection over union of two boxes."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def _average_precision(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    """
    Area under the precision–recall curve, with the precision envelope applied.

    The envelope (precision made monotonically non-increasing from the right)
    is what makes AP insensitive to the wiggle that comes from a single
    borderline detection flipping order.
    """
    if not recalls:
        return 0.0

    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    total = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            total += (mrec[i] - mrec[i - 1]) * mpre[i]
    return total


def average_precision_for_class(
    predictions: Sequence[DetBox],
    ground_truth: Sequence[DetBox],
    iou_threshold: float = 0.5,
) -> tuple[float, int, int]:
    """
    AP for a single class. Returns ``(ap, n_true_positives, n_ground_truth)``.

    Both sequences must already be filtered to one class.
    """
    n_gt = len(ground_truth)
    if n_gt == 0:
        return 0.0, 0, 0
    if not predictions:
        return 0.0, 0, n_gt

    gt_by_image: dict[str, list[DetBox]] = defaultdict(list)
    for box in ground_truth:
        gt_by_image[box.image_id].append(box)
    matched: dict[str, set[int]] = {img: set() for img in gt_by_image}

    ordered = sorted(predictions, key=lambda b: b.score, reverse=True)

    tp = [0] * len(ordered)
    fp = [0] * len(ordered)
    for i, prediction in enumerate(ordered):
        candidates = gt_by_image.get(prediction.image_id, [])
        best_iou, best_index = 0.0, -1
        for j, truth in enumerate(candidates):
            if j in matched[prediction.image_id]:
                continue
            value = iou(prediction, truth)
            if value > best_iou:
                best_iou, best_index = value, j
        if best_index >= 0 and best_iou >= iou_threshold:
            matched[prediction.image_id].add(best_index)
            tp[i] = 1
        else:
            fp[i] = 1

    cum_tp = cum_fp = 0
    recalls, precisions = [], []
    for i in range(len(ordered)):
        cum_tp += tp[i]
        cum_fp += fp[i]
        recalls.append(cum_tp / n_gt)
        precisions.append(cum_tp / max(1, cum_tp + cum_fp))

    return _average_precision(recalls, precisions), sum(tp), n_gt


def evaluate_detections(
    predictions: Iterable[DetBox],
    ground_truth: Iterable[DetBox],
    class_names: Sequence[str],
    iou_thresholds: Optional[Sequence[float]] = None,
) -> dict:
    """
    Full detection report.

    ``mAP50`` is the headline. ``mAP50_95`` averages AP over IoU 0.50–0.95 in
    steps of 0.05, which is the stricter COCO number and is the one that falls
    when boxes are roughly right but loosely placed — exactly the failure mode
    of a heatmap-derived box, so it is reported alongside rather than hidden.
    """
    predictions = list(predictions)
    ground_truth = list(ground_truth)
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]

    preds_by_class: dict[int, list[DetBox]] = defaultdict(list)
    gt_by_class: dict[int, list[DetBox]] = defaultdict(list)
    for box in predictions:
        preds_by_class[box.class_id].append(box)
    for box in ground_truth:
        gt_by_class[box.class_id].append(box)

    evaluated = sorted(gt_by_class)  # only classes that actually occur
    per_class: dict[str, dict] = {}
    ap50: list[float] = []
    ap_all: list[float] = []

    for class_id in evaluated:
        name = (
            class_names[class_id]
            if 0 <= class_id < len(class_names)
            else f"class_{class_id}"
        )
        preds = preds_by_class.get(class_id, [])
        truths = gt_by_class[class_id]

        value50, tp50, n_gt = average_precision_for_class(preds, truths, 0.5)
        ap50.append(value50)

        across = [
            average_precision_for_class(preds, truths, t)[0] for t in iou_thresholds
        ]
        mean_across = sum(across) / len(across) if across else 0.0
        ap_all.append(mean_across)

        per_class[name] = {
            "ap50": round(value50, 4),
            "ap50_95": round(mean_across, 4),
            "ground_truth": n_gt,
            "predictions": len(preds),
            "true_positives_at_50": tp50,
            "recall_at_50": round(tp50 / n_gt, 4) if n_gt else 0.0,
        }

    skipped = [
        class_names[i]
        for i in range(len(class_names))
        if i not in gt_by_class
    ]

    return {
        "mAP50": round(sum(ap50) / len(ap50), 4) if ap50 else 0.0,
        "mAP50_95": round(sum(ap_all) / len(ap_all), 4) if ap_all else 0.0,
        "classes_evaluated": len(evaluated),
        "classes_skipped_no_ground_truth": skipped,
        "total_predictions": len(predictions),
        "total_ground_truth": len(ground_truth),
        "per_class": per_class,
        "iou_thresholds": [round(t, 2) for t in iou_thresholds],
    }
