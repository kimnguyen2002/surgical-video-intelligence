"""
Weakly-supervised localisation: Grad-CAM heatmaps → boxes → real mAP.

The bridge between the platform's two sources of supervision.
:mod:`ai.training.train_tool_detection` learns instrument *presence* over all
155 surgvu24 cases from the robot installation log — a lot of supervision, but
no spatial information at all. cat1 has boxes, but only seven clips.

This module asks the question that connects them: **if a classifier trained
only on "which instruments are installed" is asked where it looked, do those
regions actually land on the instruments?** Grad-CAM produces a per-class
heatmap, the heatmap is thresholded into connected components, each component
becomes a box, and the boxes are scored against cat1's real annotations with
the same mAP implementation the supervised detector is scored with.

Run from the repository root::

    python -m ai.training.localize --checkpoint checkpoints/tool_best.pt
    python -m ai.training.localize --checkpoint ... --clips 2,4 --limit 200

What the number means
---------------------
Expect it to be *low* — considerably below a supervised detector. That is the
correct and interesting result, not a failure:

* Grad-CAM highlights whatever region drove the decision, which for an
  instrument is often the tissue interaction or the shaft rather than the
  extent a human annotator boxed.
* A classifier has no notion of instance count. Two needle drivers produce one
  blob if they overlap, and the second is scored as a miss.
* The presence labels are themselves weak: the classifier was rewarded for
  predicting instruments that were installed but off-screen, so it has learned
  to fire on context rather than only on visible evidence.

Reporting it honestly, next to the supervised number, is the point. A platform
that showed only the flattering number would be doing the thing this codebase
exists to avoid.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from ai.datasets.coco import DETECTION_CLASSES, load_all
from ai.training.config import REPO_ROOT, TOOL_CLASSES, config
from ai.training.detection_metrics import DetBox, evaluate_detections

logger = logging.getLogger("surgvu.localize")

DEFAULT_OUT = REPO_ROOT / "checkpoints" / "localization_report.json"


# ---------------------------------------------------------------------------
# Heatmap → boxes
# ---------------------------------------------------------------------------


def heatmap_to_boxes(
    heatmap,
    *,
    threshold: float = 0.4,
    min_area_fraction: float = 0.005,
    max_boxes: int = 3,
) -> list[tuple[float, float, float, float, float]]:
    """
    Convert a normalised heatmap into candidate boxes.

    Returns ``(x1, y1, x2, y2, score)`` in *heatmap* coordinates, which the
    caller scales to the frame.

    ``threshold`` is relative to the map's own maximum rather than absolute.
    Grad-CAM magnitudes vary by orders of magnitude between classes and frames,
    so a fixed cut produces everything-or-nothing; a relative one gives a
    comparable region regardless of activation scale.

    Tiny components are dropped: a handful of pixels above threshold is noise,
    and each one it emits is a false positive that costs precision.
    """
    import numpy as np  # noqa: PLC0415

    array = np.asarray(heatmap, dtype="float32")
    if array.ndim != 2 or array.size == 0:
        return []

    peak = float(array.max())
    if peak <= 0:
        return []
    normalised = array / peak
    mask = (normalised >= threshold).astype("uint8")
    if mask.sum() == 0:
        return []

    height, width = mask.shape
    min_area = max(1.0, min_area_fraction * height * width)

    components = _connected_components(mask)
    boxes: list[tuple[float, float, float, float, float]] = []
    for pixels in components:
        if len(pixels) < min_area:
            continue
        ys = [p[0] for p in pixels]
        xs = [p[1] for p in pixels]
        # Score the box by the peak activation inside it: a strong compact blob
        # should outrank a weak sprawling one when AP ranks detections.
        score = float(max(normalised[y, x] for y, x in pixels))
        boxes.append((float(min(xs)), float(min(ys)),
                      float(max(xs) + 1), float(max(ys) + 1), score))

    boxes.sort(key=lambda b: -b[4])
    return boxes[:max_boxes]


def _connected_components(mask) -> list[list[tuple[int, int]]]:
    """
    4-connected components of a binary mask.

    Uses OpenCV when available and falls back to an iterative flood fill.
    The fallback is iterative rather than recursive on purpose: a component
    spanning a 384×384 map would blow the recursion limit.
    """
    try:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        count, labels = cv2.connectedComponents(mask)
        out: list[list[tuple[int, int]]] = []
        for label in range(1, count):
            ys, xs = np.where(labels == label)
            out.append(list(zip(ys.tolist(), xs.tolist())))
        return out
    except ImportError:
        pass

    height, width = len(mask), len(mask[0])
    seen = [[False] * width for _ in range(height)]
    out = []
    for y0 in range(height):
        for x0 in range(width):
            if mask[y0][x0] == 0 or seen[y0][x0]:
                continue
            stack = [(y0, x0)]
            seen[y0][x0] = True
            pixels = []
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not seen[ny][nx]
                        and mask[ny][nx]
                    ):
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            out.append(pixels)
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    checkpoint: Path,
    *,
    clips: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    threshold: float = 0.4,
    conf_threshold: float = 0.3,
    device: str = "auto",
) -> dict:
    """
    Score Grad-CAM localisation against cat1's boxes.

    Only classes the presence model and the box vocabulary share can be
    evaluated; the rest are reported as skipped rather than silently ignored.
    """
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit("OpenCV is required: pip install opencv-python-headless") from exc

    from ai.explainability.gradcam import GradCAM, find_target_layer  # noqa: PLC0415
    from ai.training.models import load_tool_model  # noqa: PLC0415

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint}.\n"
            "Train one first:  python -m ai.training.train_tool_detection"
        )

    resolved = (
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    model, meta = load_tool_model(str(checkpoint), device=resolved)
    model.eval()

    class_names: list[str] = list(meta.get("classes") or TOOL_CLASSES)
    shared = [c for c in class_names if c in DETECTION_CLASSES]
    if not shared:
        raise SystemExit(
            "The checkpoint's classes do not overlap the cat1 box vocabulary; "
            "there is nothing to score."
        )
    logger.info("Scoring %d shared classes: %s", len(shared), ", ".join(shared))

    target_layer = find_target_layer(model, meta.get("model_name", ""))
    if target_layer is None:
        raise SystemExit("Could not resolve a Grad-CAM target layer for this model.")

    all_clips = load_all()
    if clips:
        wanted = {c.strip() for c in clips}
        all_clips = {k: v for k, v in all_clips.items() if k in wanted}
    if not all_clips:
        raise SystemExit("No cat1 clips selected.")

    size = int(config.image_size)
    mean, std = config.mean, config.std

    predictions: list[DetBox] = []
    ground_truth: list[DetBox] = []
    frames_done = 0

    cam = GradCAM(model, target_layer, multi_label=True)
    try:
        for name, clip in sorted(all_clips.items()):
            capture = cv2.VideoCapture(str(clip.video))
            if not capture.isOpened():
                logger.error("cannot open clip %s", name)
                continue

            wanted_frames = clip.frames_by_index()
            position = 0
            last = max(wanted_frames) if wanted_frames else -1

            while position <= last:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                annotation = wanted_frames.get(position)
                position += 1
                if annotation is None:
                    continue
                if limit is not None and frames_done >= limit:
                    break

                image_id = f"{name}:{annotation.index}"
                height, width = frame_bgr.shape[:2]

                for box in annotation.boxes:
                    if box.class_name not in shared:
                        continue
                    x1, y1, x2, y2 = box.xyxy()
                    ground_truth.append(
                        DetBox(image_id, DETECTION_CLASSES.index(box.class_name),
                               x1, y1, x2, y2)
                    )

                pil = Image.fromarray(frame_bgr[:, :, ::-1]).resize((size, size))
                tensor = _to_tensor(pil, mean, std, resolved)

                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.sigmoid(logits)[0]

                for class_index, class_name in enumerate(class_names):
                    if class_name not in shared:
                        continue
                    confidence = float(probs[class_index])
                    if confidence < conf_threshold:
                        continue

                    heatmap = cam.generate(tensor, class_idx=class_index)
                    if heatmap is None:
                        continue

                    for hx1, hy1, hx2, hy2, blob in heatmap_to_boxes(
                        heatmap, threshold=threshold
                    ):
                        map_h, map_w = heatmap.shape[:2]
                        sx, sy = width / map_w, height / map_h
                        predictions.append(
                            DetBox(
                                image_id,
                                DETECTION_CLASSES.index(class_name),
                                hx1 * sx, hy1 * sy, hx2 * sx, hy2 * sy,
                                # Rank by classifier confidence *and* blob
                                # strength: a confident class with a diffuse map
                                # should not outrank a confident class with a
                                # sharp one.
                                score=confidence * blob,
                            )
                        )

                frames_done += 1

            capture.release()
            if limit is not None and frames_done >= limit:
                break
    finally:
        cam.close()

    report = evaluate_detections(predictions, ground_truth, DETECTION_CLASSES)
    report["method"] = "gradcam_weakly_supervised"
    report["checkpoint"] = str(checkpoint)
    report["frames_evaluated"] = frames_done
    report["clips"] = sorted(all_clips)
    report["shared_classes"] = shared
    report["classes_in_checkpoint_without_boxes"] = [
        c for c in class_names if c not in DETECTION_CLASSES
    ]
    report["settings"] = {
        "cam_threshold": threshold,
        "confidence_threshold": conf_threshold,
        "image_size": size,
        "device": resolved,
    }
    report["interpretation"] = (
        "Localisation derived from a presence-only classifier via Grad-CAM. "
        "Expect materially lower mAP than a supervised detector: Grad-CAM marks "
        "the evidence a classifier used, which is not the same region a human "
        "annotator boxes, and a classifier cannot separate two instances of one "
        "instrument. Report alongside the supervised number, never instead of it."
    )
    return report


def _to_tensor(pil_image, mean, std, device):
    import torch  # noqa: PLC0415

    try:
        from torchvision import transforms  # noqa: PLC0415

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        return transform(pil_image).unsqueeze(0).to(device)
    except ImportError:
        import numpy as np  # noqa: PLC0415

        array = np.asarray(pil_image, dtype="float32") / 255.0
        array = (array - np.asarray(mean)) / np.asarray(std)
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clips", type=str, default=None, help="e.g. 2,4")
    parser.add_argument("--limit", type=int, default=None, help="max frames")
    parser.add_argument("--cam-threshold", type=float, default=0.4)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s │ %(message)s")

    report = evaluate(
        args.checkpoint,
        clips=args.clips.split(",") if args.clips else None,
        limit=args.limit,
        threshold=args.cam_threshold,
        conf_threshold=args.conf_threshold,
        device=args.device,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nGrad-CAM weakly-supervised localisation")
    print(f"  frames    : {report['frames_evaluated']}")
    print(f"  clips     : {', '.join(report['clips'])}")
    print(f"  mAP@50    : {report['mAP50']:.4f}")
    print(f"  mAP@50-95 : {report['mAP50_95']:.4f}")
    print("\n  per class:")
    for name, stats in report["per_class"].items():
        print(
            f"    {name:32s} AP50 {stats['ap50']:.3f}  "
            f"recall {stats['recall_at_50']:.3f}  (gt {stats['ground_truth']})"
        )
    print(f"\n  report → {args.out}")
    print(f"\n  {report['interpretation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
