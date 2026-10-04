#!/usr/bin/env python3
"""
Measure what the live loop will cost, and check the models against the labels.

Two questions, and the second one is the one that matters:

**Latency** — how long does one frame take through each graph. Native ORT on
this machine is the floor; onnxruntime-web is roughly 1.5-3x slower on the WASM
backend and comparable on WebGPU, so treat these as optimistic.

**Agreement** — the app is about to draw these predictions next to recorded
dataset labels, so it had better be the case that they broadly agree. Frames
are sampled at timestamps where `tasks.csv` and `tools.csv` say something
specific, and the model is asked the same question.

A caveat the numbers cannot carry on their own: the task checkpoint's
`train_cases` include 000-005, which are the only cases with video here. So
agreement on them is **not** held-out accuracy — it is a check that
preprocessing, class order, and the export are wired up correctly. The
checkpoint's own held-out number (balanced accuracy 0.776 over 20 unseen
cases) is the one to quote.

    python3 scripts/benchmark_models.py --frames 60
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
CHARLIE = HERE.parent
MODELS = CHARLIE / "public" / "models"
VIDEOS = CHARLIE.parent / "surgvu24_videos_only"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing — must match ai/training/dataset.py:default_transform(False)
# ---------------------------------------------------------------------------


def preprocess_classifier(frame: np.ndarray, size: int) -> np.ndarray:
    """
    Mirror `default_transform(train=False)` exactly.

    torchvision's `Resize(int)` scales the *shorter* side and keeps the aspect
    ratio; `CenterCrop` then takes the middle square. Squashing a 16:9 frame
    into a square instead — the obvious thing to write — feeds the model a
    horizontally compressed image it never saw in training.
    """
    h, w = frame.shape[:2]
    scale = int(size * 1.14) / min(h, w)
    resized = cv2.resize(
        frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR
    )
    rh, rw = resized.shape[:2]
    top, left = (rh - size) // 2, (rw - size) // 2
    cropped = resized[top : top + size, left : left + size]

    x = cropped.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(x, (2, 0, 1))[None]


def preprocess_yolo(frame: np.ndarray, size: int) -> np.ndarray:
    """
    Ultralytics letterbox: scale to fit, pad the remainder with grey.

    Unlike the classifiers this keeps the whole frame — an instrument entering
    from the edge is exactly what the detector is for.
    """
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = cv2.resize(
        frame, (nw, nh), interpolation=cv2.INTER_LINEAR
    )
    x = canvas.astype(np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))[None]


# ---------------------------------------------------------------------------
# Labels — read back the data the app actually ships
# ---------------------------------------------------------------------------


def load_bundled_labels() -> tuple[list[dict], list[dict], list[dict]]:
    text = (CHARLIE / "src" / "data" / "surgvuLabels.ts").read_text(encoding="utf-8")
    tools = json.loads(
        re.search(r"SURGVU_TOOL_INTERVALS[^=]*= (\[.*?\]);\n\nexport const", text, re.S).group(1)
    )
    tasks = json.loads(
        re.search(r"SURGVU_TASK_INTERVALS[^=]*= (\[.*?\]);\n$", text, re.S).group(1)
    )
    cases_text = (CHARLIE / "src" / "data" / "surgvuCases.ts").read_text(encoding="utf-8")
    cases = json.loads(
        re.search(r"SURGVU_CASES: SurgvuCase\[\] = (\[.*?\]);\n\n/\*\*", cases_text, re.S).group(1)
    )
    return tools, tasks, cases


def video_path(cases: list[dict], case_id: str, part: int) -> Path | None:
    for case in cases:
        if case["caseId"] != case_id:
            continue
        for p in case["parts"]:
            if p["part"] == part:
                return VIDEOS / case["dirName"] / p["filename"]
    return None


def grab(path: Path, seconds: float) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = capture.read()
    capture.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def session_for(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def timed(session: ort.InferenceSession, feed: dict, runs: int) -> tuple[float, float]:
    for _ in range(5):
        session.run(None, feed)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def decode_yolo(output: np.ndarray, conf: float = 0.25) -> int:
    pred = output[0].T
    return int((pred[:, 4:].max(axis=1) > conf).sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    manifest = json.loads((MODELS / "manifest.json").read_text())
    entries = {m["name"]: m for m in manifest["models"]}
    tool_intervals, task_intervals, cases = load_bundled_labels()

    # ---------------- latency ----------------
    print("=" * 68)
    print("LATENCY  (native onnxruntime, CPU, %d threads)" % args.threads)
    print("=" * 68)

    probe = np.zeros((720, 1280, 3), dtype=np.uint8)
    sessions = {}
    total_median = 0.0
    for name, entry in entries.items():
        path = MODELS / entry["file"]
        session = session_for(path, args.threads)
        sessions[name] = session
        prep = preprocess_yolo if name == "detection" else preprocess_classifier
        feed = {session.get_inputs()[0].name: prep(probe, entry["imageSize"])}
        median, p95 = timed(session, feed, args.runs)
        total_median += median
        print(
            f"  {name:<10} {entry['imageSize']:>4}px  "
            f"{path.stat().st_size / 1e6:>5.1f} MB  "
            f"{median:>6.1f} ms median  {p95:>6.1f} ms p95"
        )
    print(f"  {'all three':<10}                      {total_median:>6.1f} ms")
    print(
        f"  {'detector only':<10}                  "
        f"{timed(sessions['detection'], {sessions['detection'].get_inputs()[0].name: preprocess_yolo(probe, 448)}, args.runs)[0]:>6.1f} ms"
        "   <- the per-frame cost in the live loop"
    )

    # ---------------- task agreement ----------------
    print()
    print("=" * 68)
    print("TASK CLASSIFIER vs tasks.csv")
    print("  NOTE: cases 000-005 are in this checkpoint's train_cases, so this")
    print("  is a wiring check, not held-out accuracy (that is 0.776 balanced).")
    print("=" * 68)

    task_entry = entries["task"]
    task_session = sessions["task"]
    task_in = task_session.get_inputs()[0].name
    task_classes = task_entry["classes"]

    sampled = task_intervals[:: max(1, len(task_intervals) // args.frames)][: args.frames]
    hits = 0
    checked = 0
    confusion: Counter = Counter()

    for interval in sampled:
        path = video_path(cases, interval["case"], interval["part"])
        if not path or not path.exists():
            continue
        frame = grab(path, (interval["start"] + interval["end"]) / 2)
        if frame is None:
            continue
        logits = task_session.run(None, {task_in: preprocess_classifier(frame, task_entry["imageSize"])})[0]
        predicted = task_classes[int(logits.argmax())]
        checked += 1
        if predicted == interval["label"]:
            hits += 1
        else:
            confusion[f"{interval['label']} -> {predicted}"] += 1

    if checked:
        print(f"  {hits}/{checked} frames matched the recorded task ({100 * hits / checked:.0f}%)")
        for pair, n in confusion.most_common(5):
            print(f"    missed: {pair}  ({n}x)")

    # ---------------- tool agreement ----------------
    print()
    print("=" * 68)
    print("TOOL PRESENCE vs tools.csv, and DETECTOR box counts")
    print("=" * 68)

    tool_entry = entries["tool"]
    tool_session = sessions["tool"]
    tool_in = tool_session.get_inputs()[0].name
    tool_classes = tool_entry["classes"]
    det_session = sessions["detection"]
    det_in = det_session.get_inputs()[0].name

    sampled = tool_intervals[:: max(1, len(tool_intervals) // args.frames)][: args.frames]
    any_hit = 0
    checked = 0
    boxes_total = 0

    for interval in sampled:
        path = video_path(cases, interval["case"], interval["part"])
        if not path or not path.exists():
            continue
        frame = grab(path, (interval["start"] + interval["end"]) / 2)
        if frame is None:
            continue

        logits = tool_session.run(None, {tool_in: preprocess_classifier(frame, tool_entry["imageSize"])})[0]
        probs = 1 / (1 + np.exp(-logits[0]))
        predicted = {tool_classes[i] for i in np.where(probs > 0.5)[0]}
        checked += 1
        if interval["label"] in predicted:
            any_hit += 1

        boxes_total += decode_yolo(
            det_session.run(None, {det_in: preprocess_yolo(frame, entries["detection"]["imageSize"])})[0]
        )

    if checked:
        print(f"  recorded instrument also predicted present: {any_hit}/{checked} ({100 * any_hit / checked:.0f}%)")
        print(f"  detector boxes above 0.25 conf: {boxes_total / checked:.1f} per frame on average")
        print()
        print("  Recorded presence comes from the robot's installation log, so an")
        print("  instrument counts as present while off-screen or occluded. A miss")
        print("  here is often the model being visually right and the label being")
        print("  about the arm, not the picture.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
