"""
Extract labelled frames for surgical **task** recognition, by seeking.

Run from the repository root::

    python -m ai.training.extract_task_frames --cases 16 --per-interval 12
    python -m ai.training.extract_task_frames --cases 4 --per-interval 4   # quick

Why seeking rather than decoding
--------------------------------
`extract_frames` walks a video front to back, calling ``grab()`` on every one
of ~1 million frames. Measured here, that is ~11 minutes for a single 4-hour
part — roughly a day for the 144 parts on disk, and most of that work is thrown
away.

Task recognition does not need the whole video. ``tasks.csv`` says exactly when
each labelled activity happens, and those intervals cover only about a third of
a case. Seeking straight to sampled timestamps inside each interval reads two
orders of magnitude fewer frames, and it produces a *balanced* dataset directly:
frames are drawn per interval rather than per second, so a 33-minute suturing
run does not drown out a 2-minute range-of-motion run.

Sampling
--------
Frames are taken at even offsets **inside** each interval, with a margin at
both ends. Task boundaries in SurgVU are coarse, and a frame taken at the exact
start of an interval is as likely to show the previous activity as the labelled
one — training on those teaches the boundary noise rather than the task.

Labels come from the corrected parser in :mod:`ai.training.labels`, so intervals
that span video parts resolve properly. This is the pipeline the plural/singular
and part-index fixes exist for.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from ai.datasets.corpora import registry
from ai.training.config import REPO_ROOT, TASK_CLASSES, TASK_DISPLAY
from ai.training.labels import load_label_rows, resolve_intervals

logger = logging.getLogger("surgvu.task_frames")

DEFAULT_OUT = REPO_ROOT / "data" / "surgvu_tasks"

#: Skip this fraction of an interval at each end before sampling.
BOUNDARY_MARGIN = 0.12
#: Intervals shorter than this are dropped — too short to be a stable label.
MIN_INTERVAL_S = 8.0


def _probe_durations(case: str) -> dict[tuple[str, int], float]:
    from ai.datasets.groundtruth import part_durations_for_case

    return dict(part_durations_for_case(case))


def sample_timestamps(start: float, end: float, count: int) -> list[float]:
    """Evenly spaced timestamps inside an interval, avoiding both boundaries."""
    span = end - start
    if span <= 0:
        return []
    margin = span * BOUNDARY_MARGIN
    lo, hi = start + margin, end - margin
    if hi <= lo:
        return [(start + end) / 2.0]
    if count <= 1:
        return [(lo + hi) / 2.0]
    step = (hi - lo) / (count - 1)
    return [lo + step * i for i in range(count)]


def extract_case(
    case: str,
    out_root: Path,
    *,
    per_interval: int,
    quality: int = 88,
    max_side: int = 512,
) -> list[dict]:
    """Extract every labelled task frame for one case. Returns manifest rows."""
    import cv2

    record = registry.surgvu_cases.get(case)
    if record is None or not record.tasks_csv or not record.parts:
        return []

    durations = _probe_durations(case)
    if not durations:
        logger.warning("case %s: no probeable video parts", case)
        return []

    try:
        rows = load_label_rows(record.tasks_csv, "task", case_id=case)
    except Exception as exc:
        logger.error("case %s: tasks.csv failed: %s", case, exc)
        return []

    intervals = [
        i for i in resolve_intervals(rows, durations)
        if (i.end - i.start) >= MIN_INTERVAL_S and i.label in TASK_CLASSES
    ]
    if not intervals:
        return []

    by_part: dict[int, list] = defaultdict(list)
    for interval in intervals:
        by_part[interval.part].append(interval)

    manifest: list[dict] = []
    for part, part_intervals in sorted(by_part.items()):
        video = record.parts.get(part)
        if video is None:
            continue

        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            logger.error("cannot open %s", video)
            continue

        try:
            for interval in sorted(part_intervals, key=lambda i: i.start):
                out_dir = out_root / interval.label
                out_dir.mkdir(parents=True, exist_ok=True)

                for timestamp in sample_timestamps(
                    interval.start, interval.end, per_interval
                ):
                    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        continue

                    height, width = frame.shape[:2]
                    if max(height, width) > max_side:
                        scale = max_side / max(height, width)
                        frame = cv2.resize(
                            frame, (int(width * scale), int(height * scale)),
                            interpolation=cv2.INTER_AREA,
                        )

                    name = f"case{case}_p{part}_{int(timestamp):06d}.jpg"
                    path = out_dir / name
                    cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                    manifest.append(
                        {
                            "file": str(path.relative_to(out_root)).replace("\\", "/"),
                            "case": case,
                            "part": part,
                            "timestamp": round(timestamp, 2),
                            "label": interval.label,
                        }
                    )
        finally:
            capture.release()

    logger.info("case %s → %d frames from %d intervals", case, len(manifest), len(intervals))
    return manifest


def extract(
    out_root: Path = DEFAULT_OUT,
    *,
    cases: Optional[int] = None,
    per_interval: int = 12,
    seed: int = 42,
    case_ids: Optional[list[str]] = None,
) -> dict:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    available = [
        c for c, record in registry.surgvu_cases.items()
        if record.parts and record.tasks_csv
    ]
    if not available:
        raise SystemExit(
            "No surgvu24 cases with both video and tasks.csv were found.\n"
            "Check SURGVU_VIDEO_ROOT / SURGVU_LABEL_ROOT."
        )

    if case_ids:
        chosen = [c for c in case_ids if c in available]
    else:
        # Deterministic spread across the corpus rather than the first N,
        # which would all come from one contiguous block of case numbers.
        rng = random.Random(seed)
        shuffled = sorted(available)
        rng.shuffle(shuffled)
        chosen = sorted(shuffled[: cases or len(shuffled)])

    logger.info("extracting %d cases: %s", len(chosen), ", ".join(chosen))

    manifest: list[dict] = []
    for index, case in enumerate(chosen, start=1):
        logger.info("[%d/%d] case %s", index, len(chosen), case)
        manifest.extend(extract_case(case, out_root, per_interval=per_interval))

    counts = Counter(row["label"] for row in manifest)
    payload = {
        "frames": len(manifest),
        "cases": chosen,
        "per_interval": per_interval,
        "classes": TASK_CLASSES,
        "counts": {k: counts.get(k, 0) for k in TASK_CLASSES},
        "manifest": manifest,
    }
    (out_root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cases", type=int, default=16, help="how many cases to sample")
    parser.add_argument("--case-ids", type=str, default=None, help="explicit list, comma-separated")
    parser.add_argument("--per-interval", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s │ %(message)s")

    payload = extract(
        args.out,
        cases=args.cases,
        per_interval=args.per_interval,
        seed=args.seed,
        case_ids=args.case_ids.split(",") if args.case_ids else None,
    )

    print(f"\nExtracted {payload['frames']} task frames from {len(payload['cases'])} cases")
    print(f"  output: {args.out}")
    print("\n  per class:")
    for name in TASK_CLASSES:
        count = payload["counts"].get(name, 0)
        flag = "" if count else "   ← none found"
        print(f"    {TASK_DISPLAY.get(name, name):36s} {count:6d}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
