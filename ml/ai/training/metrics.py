"""
Evaluation metrics, implemented directly so results do not depend on whether
scikit-learn happens to be installed.

For multi-label tool detection, **mean average precision is the headline
metric, not accuracy**. With 12 classes where most are absent in most frames,
a model that always predicts "absent" scores above 90% accuracy and is
worthless. Average precision integrates precision over the full recall range
and is unmoved by that degenerate solution.

Per-class numbers are always reported alongside the mean, because a good macro
average routinely hides a rare class the model never gets right.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Multi-label
# ---------------------------------------------------------------------------
def average_precision(scores: Sequence[float], targets: Sequence[int]) -> float:
    """
    Area under the precision-recall curve for one class.

    Uses the "all-points" interpolation (the standard for VOC-style mAP):
    precision is accumulated at every positive as the threshold sweeps down.
    """
    pairs = sorted(zip(scores, targets), key=lambda p: -p[0])
    positives = sum(targets)
    if positives == 0:
        return float("nan")  # Undefined, not zero — excluded from the mean.

    hits = 0
    running = 0.0
    for i, (_, target) in enumerate(pairs, start=1):
        if target:
            hits += 1
            running += hits / i
    return running / positives


def mean_average_precision(
    scores: Sequence[Sequence[float]], targets: Sequence[Sequence[int]]
) -> tuple[float, list[float]]:
    """mAP over classes, skipping classes with no positive examples."""
    if not scores:
        return float("nan"), []
    num_classes = len(scores[0])
    per_class = []
    for c in range(num_classes):
        per_class.append(
            average_precision([s[c] for s in scores], [int(t[c]) for t in targets])
        )
    valid = [ap for ap in per_class if not math.isnan(ap)]
    return (sum(valid) / len(valid) if valid else float("nan")), per_class


def binary_counts(
    scores: Sequence[float], targets: Sequence[int], threshold: float = 0.5
) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for score, target in zip(scores, targets):
        predicted = score >= threshold
        if predicted and target:
            tp += 1
        elif predicted and not target:
            fp += 1
        elif not predicted and target:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def precision_recall_f1(
    scores: Sequence[float], targets: Sequence[int], threshold: float = 0.5
) -> dict:
    tp, fp, tn, fn = binary_counts(scores, targets, threshold)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": tp + fn,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def best_threshold(scores: Sequence[float], targets: Sequence[int]) -> tuple[float, float]:
    """
    The threshold maximising F1 for one class.

    A single global 0.5 is rarely right for every class when prevalence spans
    orders of magnitude; per-class thresholds tuned on validation (never test)
    typically move rare-class F1 substantially.
    """
    best_f1, best_t = 0.0, 0.5
    for step in range(5, 96, 5):
        threshold = step / 100
        f1 = precision_recall_f1(scores, targets, threshold)["f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, threshold
    return best_t, round(best_f1, 4)


def multilabel_report(
    scores: Sequence[Sequence[float]],
    targets: Sequence[Sequence[int]],
    class_names: Sequence[str],
    threshold: float = 0.5,
    tune_thresholds: bool = False,
) -> dict:
    """Full multi-label evaluation report."""
    m_ap, per_class_ap = mean_average_precision(scores, targets)

    classes = {}
    f1s, thresholds = [], {}
    for c, name in enumerate(class_names):
        column_scores = [s[c] for s in scores]
        column_targets = [int(t[c]) for t in targets]

        class_threshold = threshold
        if tune_thresholds and any(column_targets):
            class_threshold, _ = best_threshold(column_scores, column_targets)
        thresholds[name] = class_threshold

        stats = precision_recall_f1(column_scores, column_targets, class_threshold)
        stats["average_precision"] = (
            None if math.isnan(per_class_ap[c]) else round(per_class_ap[c], 4)
        )
        stats["threshold"] = class_threshold
        stats["prevalence"] = round(sum(column_targets) / max(len(column_targets), 1), 4)
        classes[name] = stats
        if stats["support"]:
            f1s.append(stats["f1"])

    # Micro-average: pool every (sample, class) decision.
    flat_scores = [s[c] for s in scores for c in range(len(class_names))]
    flat_targets = [int(t[c]) for t in targets for c in range(len(class_names))]
    micro = precision_recall_f1(flat_scores, flat_targets, threshold)

    exact = sum(
        1
        for score, target in zip(scores, targets)
        if all((s >= thresholds[n]) == bool(t) for s, t, n in zip(score, target, class_names))
    )

    return {
        "samples": len(scores),
        "mAP": None if math.isnan(m_ap) else round(m_ap, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "micro_f1": micro["f1"],
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        # Every class correct on a frame — a strict, informative measure.
        "exact_match": round(exact / max(len(scores), 1), 4),
        "per_class": classes,
    }


# ---------------------------------------------------------------------------
# Multi-class
# ---------------------------------------------------------------------------
def confusion_matrix(
    predictions: Sequence[int], targets: Sequence[int], num_classes: int
) -> list[list[int]]:
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for predicted, target in zip(predictions, targets):
        if 0 <= target < num_classes and 0 <= predicted < num_classes:
            matrix[target][predicted] += 1
    return matrix


def multiclass_report(
    predictions: Sequence[int],
    targets: Sequence[int],
    class_names: Sequence[str],
    probabilities: Optional[Sequence[Sequence[float]]] = None,
) -> dict:
    num_classes = len(class_names)
    matrix = confusion_matrix(predictions, targets, num_classes)
    total = len(targets) or 1

    classes = {}
    f1s = []
    for c, name in enumerate(class_names):
        tp = matrix[c][c]
        fp = sum(matrix[r][c] for r in range(num_classes)) - tp
        fn = sum(matrix[c]) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = sum(matrix[c])
        classes[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        if support:
            f1s.append(f1)

    accuracy = sum(matrix[c][c] for c in range(num_classes)) / total

    report = {
        "samples": len(targets),
        "accuracy": round(accuracy, 4),
        # Balanced accuracy is the honest headline when classes are skewed:
        # 'other' dominates SurgVU tasks and inflates plain accuracy.
        "balanced_accuracy": round(
            sum(classes[n]["recall"] for n in class_names if classes[n]["support"])
            / max(sum(1 for n in class_names if classes[n]["support"]), 1),
            4,
        ),
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "confusion_matrix": matrix,
        "labels": list(class_names),
        "per_class": classes,
    }

    if probabilities:
        report["top2_accuracy"] = round(_top_k_accuracy(probabilities, targets, 2), 4)
    return report


def _top_k_accuracy(
    probabilities: Sequence[Sequence[float]], targets: Sequence[int], k: int
) -> float:
    hits = 0
    for probability, target in zip(probabilities, targets):
        ranked = sorted(range(len(probability)), key=lambda i: -probability[i])[:k]
        hits += int(target in ranked)
    return hits / max(len(targets), 1)


def format_report(report: dict, title: str = "Evaluation") -> str:
    """Render a report as a fixed-width table for logs and stdout."""
    lines = [f"\n{'=' * 78}", f"  {title}", "=" * 78]

    headline = [
        f"{key}: {report[key]}"
        for key in ("samples", "mAP", "macro_f1", "micro_f1", "accuracy", "balanced_accuracy", "exact_match")
        if key in report and report[key] is not None
    ]
    lines.append("  " + "   ".join(headline))
    lines.append("-" * 78)
    lines.append(f"  {'class':<34}{'prec':>8}{'rec':>8}{'f1':>8}{'AP':>8}{'n':>10}")
    lines.append("-" * 78)

    for name, stats in report.get("per_class", {}).items():
        ap = stats.get("average_precision")
        lines.append(
            f"  {name[:33]:<34}{stats['precision']:>8.3f}{stats['recall']:>8.3f}"
            f"{stats['f1']:>8.3f}{(ap if ap is not None else float('nan')):>8.3f}"
            f"{stats['support']:>10}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)
