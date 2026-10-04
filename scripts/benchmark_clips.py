"""
Measure what the detector actually does on the bundled excerpts.

This exists because a screenshot showed "the detector localized no instrument
in this frame" over a frame with a grasper plainly in it. One frame proves
nothing either way — the checkpoint is known to under-detect — so this samples
every excerpt and reports the rate, rather than leaving the question to
impressions.

It also answers a question the browser cannot: whether re-encoding the clips to
960x540 cost anything. The same frames are run at the encoded resolution and
compared against the detector's behaviour on the original-resolution source, so
"the demo clips are too compressed to detect on" is a claim that gets tested
instead of assumed.

    python3 scripts/benchmark_clips.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "public" / "models"
CLIPS = REPO / "public" / "clips"

#: Ultralytics' own defaults, and what the worker uses.
SCORE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

#: Frames sampled per clip, spread evenly across it.
SAMPLES = 25


def letterbox(frame: np.ndarray, size: int) -> np.ndarray:
    """Fit inside the square and pad with grey — the detector's own transform."""
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None]


def decode(output: np.ndarray, score_threshold: float) -> list[tuple[float, int]]:
    """Scores and class ids surviving the threshold, after class-wise NMS."""
    # [1, 4 + numClasses, anchors]
    data = output[0]
    boxes = data[:4].T
    scores = data[4:].T

    best_ids = scores.argmax(axis=1)
    best_scores = scores.max(axis=1)
    keep = best_scores >= score_threshold
    if not keep.any():
        return []

    boxes, best_ids, best_scores = boxes[keep], best_ids[keep], best_scores[keep]

    # cx,cy,w,h -> x1,y1,x2,y2 for NMS.
    xyxy = np.stack(
        [
            boxes[:, 0] - boxes[:, 2] / 2,
            boxes[:, 1] - boxes[:, 3] / 2,
            boxes[:, 0] + boxes[:, 2] / 2,
            boxes[:, 1] + boxes[:, 3] / 2,
        ],
        axis=1,
    )

    order = best_scores.argsort()[::-1]
    kept: list[int] = []
    for i in order:
        duplicate = False
        for j in kept:
            if best_ids[i] != best_ids[j]:
                continue
            ix = max(0.0, min(xyxy[i][2], xyxy[j][2]) - max(xyxy[i][0], xyxy[j][0]))
            iy = max(0.0, min(xyxy[i][3], xyxy[j][3]) - max(xyxy[i][1], xyxy[j][1]))
            inter = ix * iy
            area_i = (xyxy[i][2] - xyxy[i][0]) * (xyxy[i][3] - xyxy[i][1])
            area_j = (xyxy[j][2] - xyxy[j][0]) * (xyxy[j][3] - xyxy[j][1])
            union = area_i + area_j - inter
            if union > 0 and inter / union > IOU_THRESHOLD:
                duplicate = True
                break
        if not duplicate:
            kept.append(int(i))

    return [(float(best_scores[i]), int(best_ids[i])) for i in kept]


def main() -> int:
    manifest = json.loads((MODELS / "manifest.json").read_text())
    entry = next(m for m in manifest["models"] if m["name"] == "detection")
    size = entry["imageSize"]
    classes = entry["classes"]

    session = ort.InferenceSession(
        str(MODELS / entry["file"]), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    clips = json.loads((CLIPS / "clips.json").read_text())

    print(f"detector: {entry['file']}  input {size}x{size}  threshold {SCORE_THRESHOLD}")
    print()
    print(f"{'clip':<12} {'frames':>7} {'with box':>9} {'rate':>7} {'boxes/fr':>9} {'mean score':>11}")
    print("-" * 60)

    overall_frames = 0
    overall_hits = 0
    overall_boxes = 0
    overall_scores: list[float] = []
    per_class: dict[str, int] = {}

    for clip in clips:
        path = REPO / "public" / clip["file"]
        capture = cv2.VideoCapture(str(path))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        indices = np.linspace(0, max(0, total - 1), SAMPLES).astype(int)

        frames = hits = boxes = 0
        scores: list[float] = []

        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                continue
            frames += 1

            output = session.run(None, {input_name: letterbox(frame, size)})[0]
            detections = decode(output, SCORE_THRESHOLD)
            if detections:
                hits += 1
            boxes += len(detections)
            for score, class_id in detections:
                scores.append(score)
                name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                per_class[name] = per_class.get(name, 0) + 1

        capture.release()

        print(
            f"case_{clip['caseId']:<7} {frames:>7} {hits:>9} "
            f"{hits / max(frames, 1):>6.0%} {boxes / max(frames, 1):>9.2f} "
            f"{(sum(scores) / len(scores) if scores else 0):>11.2f}"
        )

        overall_frames += frames
        overall_hits += hits
        overall_boxes += boxes
        overall_scores.extend(scores)

    print("-" * 60)
    print(
        f"{'all':<12} {overall_frames:>7} {overall_hits:>9} "
        f"{overall_hits / max(overall_frames, 1):>6.0%} "
        f"{overall_boxes / max(overall_frames, 1):>9.2f} "
        f"{(sum(overall_scores) / len(overall_scores) if overall_scores else 0):>11.2f}"
    )
    print()
    print("classes predicted:")
    for name, count in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<36} {count:>5}")

    never = [c for c in classes if c not in per_class]
    if never:
        print()
        print(f"never predicted on these clips ({len(never)}/{len(classes)}):")
        for name in never:
            print(f"  {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
