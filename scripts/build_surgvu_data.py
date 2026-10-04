#!/usr/bin/env python3
"""
Generate charlie's bundled SurgVU ground truth from the local dataset.

Charlie runs entirely in the browser on AI Studio, so it cannot read
``tools.csv`` at request time and it certainly cannot ship 8.8 GB of video.
What it *can* ship is the labels — the whole annotation set for the six cases
present locally is a few hundred rows, which compiles to well under a
megabyte of TypeScript.

Two things are emitted:

``src/data/surgvuCases.ts``    the part manifest: which files exist, how long
                              each one actually runs, and its resolution.
``src/data/surgvuLabels.ts``   every labelled interval, already resolved to a
                              single video part.

Durations matter and are not guessable. ``tools.csv`` describes intervals as
``(part, seconds)`` and a row may open in part 1 and close in part 2; splitting
that row across parts requires knowing how long part 1 runs. Durations are read
from the video containers with OpenCV.

Parsing is delegated to ``ai/training/labels.py`` in surgical_main so that the
browser app and the training pipeline agree on exactly one normalisation of
every label string. Run this again after adding video cases:

    python3 scripts/build_surgvu_data.py
"""

from __future__ import annotations

import csv
import json
import sys
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORKSPACE = REPO.parent
MAIN = REPO / "ml"

# The release is not in this repository and never will be: ~168 GB under its
# own data-use terms. These point at wherever it was unpacked, with the
# environment taking precedence so nobody has to edit this file.
VIDEO_ROOT = Path(
    os.environ.get("SURGVU_VIDEO_ROOT", "")
) or next(
    (WORKSPACE / name for name in ("surgvu_videos", "surgvu24_videos_only")
     if (WORKSPACE / name).is_dir()),
    WORKSPACE / "surgvu24_videos_only",
)
LABEL_ROOT = Path(
    os.environ.get("SURGVU_LABEL_ROOT", "")
) or (WORKSPACE / "surgvu24_labels_updated_v2" / "labels")

sys.path.insert(0, str(REPO))

from ai.training.labels import (  # noqa: E402
    _DROP_TOOL_LABELS,
    _resolve_columns,
    case_id_from_path,
    normalise_label,
    parse_case_part,
    parse_time,
    resolve_intervals,
)
from ai.training.config import TASK_DISPLAY, TOOL_DISPLAY  # noqa: E402


def probe_parts() -> list[dict]:
    """Read every case directory's video parts and their true durations."""
    import cv2

    cases: list[dict] = []
    for case_dir in sorted(VIDEO_ROOT.glob("case_*")):
        if not case_dir.is_dir():
            continue
        case_id = case_id_from_path(case_dir / "x")
        parts = []
        for video in sorted(case_dir.glob("*.mp4")):
            capture = cv2.VideoCapture(str(video))
            frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv2.CAP_PROP_FPS)
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            capture.release()
            if not fps or not frames:
                print(f"  !! could not probe {video.name}, skipping")
                continue
            _, part = parse_case_part(video.stem, case_id)
            parts.append(
                {
                    "part": part,
                    "filename": video.name,
                    "durationSeconds": round(float(frames) / float(fps), 3),
                    "fps": round(float(fps), 3),
                    "width": int(width),
                    "height": int(height),
                    "sizeBytes": video.stat().st_size,
                }
            )
        if parts:
            cases.append({"caseId": case_id, "dirName": case_dir.name, "parts": parts})
            total = sum(p["durationSeconds"] for p in parts)
            print(f"  case_{case_id}: {len(parts)} part(s), {total / 60:.1f} min")
    return cases


def read_rows(csv_path: Path, kind: str) -> list[dict]:
    """
    Parse one label CSV, keeping the columns ``labels.py`` discards.

    ``labels.py`` drops ``commercial_toolname`` because training only needs the
    class. The interface shows it — "SureForm Stapler 60" is what the surgeon
    calls the thing on the screen — so it is carried through here.
    """
    case_id = case_id_from_path(csv_path)
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = _resolve_columns(reader.fieldnames or [])
        commercial_col = next(
            (n for n in (reader.fieldnames or []) if "commercial" in n.strip().lower()),
            None,
        )
        label_column = columns[kind]

        rows = []
        for row in reader:
            label = normalise_label(row.get(label_column, ""))
            if not label:
                continue
            if kind == "tool" and label in _DROP_TOOL_LABELS:
                continue

            install_case, install_part = parse_case_part(
                row.get(columns["install_part"], ""), case_id
            )
            uninstall_case, uninstall_part = install_case, install_part
            uninstall_time = parse_time(row.get(columns["install_time"], ""))
            if columns.get("uninstall_part") and row.get(columns["uninstall_part"]):
                uninstall_case, uninstall_part = parse_case_part(
                    row[columns["uninstall_part"]], case_id
                )
                uninstall_time = parse_time(row.get(columns.get("uninstall_time", ""), ""))

            rows.append(
                {
                    "case": install_case,
                    "start_part": install_part,
                    "start_time": parse_time(row.get(columns["install_time"], "")),
                    "end_case": uninstall_case,
                    "end_part": uninstall_part,
                    "end_time": uninstall_time,
                    "arm": (row.get(columns.get("arm", ""), "") or "").strip(),
                    "label": label,
                    "commercial": (row.get(commercial_col, "") or "").strip()
                    if commercial_col
                    else "",
                }
            )
    return rows


def build_intervals(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """Resolve both label kinds against the probed part durations."""
    durations = {
        (case["caseId"], part["part"]): part["durationSeconds"]
        for case in cases
        for part in case["parts"]
    }

    tools: list[dict] = []
    tasks: list[dict] = []

    for case in cases:
        case_dir = LABEL_ROOT / case["dirName"]
        if not case_dir.is_dir():
            print(f"  !! no labels for {case['dirName']}")
            continue

        for kind, filename, sink, display in (
            ("tool", "tools.csv", tools, TOOL_DISPLAY),
            ("task", "tasks.csv", tasks, TASK_DISPLAY),
        ):
            csv_path = case_dir / filename
            if not csv_path.exists():
                continue

            rows = read_rows(csv_path, kind)
            # resolve_intervals splits rows that span parts; it reads only the
            # keys labels.py defines, so the extra columns ride along in a
            # parallel lookup keyed by the row's identity.
            extras = {
                (r["case"], r["start_part"], r["start_time"], r["label"]): r
                for r in rows
            }

            for interval in resolve_intervals(rows, durations):
                extra = extras.get(
                    (interval.case, interval.part, interval.start, interval.label), {}
                )
                record = {
                    "case": interval.case,
                    "part": interval.part,
                    "start": round(interval.start, 2),
                    "end": round(interval.end, 2),
                    "label": interval.label,
                    "display": display.get(
                        interval.label, interval.label.replace("_", " ")
                    ),
                }
                if kind == "tool":
                    record["arm"] = interval.arm
                    record["commercial"] = extra.get("commercial", "")
                sink.append(record)

    return dedupe(tools), dedupe(tasks)


def dedupe(records: list[dict]) -> list[dict]:
    """
    Collapse rows the release repeats verbatim.

    The v2 CSVs contain exact duplicates — case_002 alone has ~20 rows whose
    only difference from the row above is the ``index`` column. Training never
    noticed, because ``IntervalIndex`` reduces overlapping intervals to a *set*
    of active labels. A per-arm ribbon does notice: it would draw the same bar
    twice and report a tool count that is too high. Two identical intervals
    assert one fact, so they become one record here.
    """
    seen: set[tuple] = set()
    unique = []
    for record in records:
        key = (
            record["case"],
            record["part"],
            record["start"],
            record["end"],
            record["label"],
            record.get("arm", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)

    unique.sort(key=lambda r: (r["case"], r["part"], r["start"], r.get("arm", "")))
    dropped = len(records) - len(unique)
    if dropped:
        print(f"  collapsed {dropped} duplicate interval(s) from the release")
    return unique


BANNER = """/**
 * GENERATED FILE — do not edit by hand.
 *
 * Produced by scripts/build_surgvu_data.py from the SurgVU 2024 release:
 *   videos  surgvu24_videos_only/
 *   labels  surgvu24_labels_updated_v2/labels/
 *
 * Regenerate with:  python3 scripts/build_surgvu_data.py
 */
"""


def emit_cases(cases: list[dict], tools: list[dict], tasks: list[dict]) -> str:
    for case in cases:
        cid = case["caseId"]
        case["toolClasses"] = sorted({t["label"] for t in tools if t["case"] == cid})
        case["taskClasses"] = sorted({t["label"] for t in tasks if t["case"] == cid})
        case["totalDurationSeconds"] = round(
            sum(p["durationSeconds"] for p in case["parts"]), 3
        )
        case["toolIntervalCount"] = sum(1 for t in tools if t["case"] == cid)
        case["taskIntervalCount"] = sum(1 for t in tasks if t["case"] == cid)

    return f"""{BANNER}
export interface SurgvuPart {{
  /** Part index as the label CSVs number it — 1-based, not 0-based. */
  part: number;
  /** Exact filename on disk; the file picker matches against this. */
  filename: string;
  /** True container duration in seconds, read from the video itself. */
  durationSeconds: number;
  fps: number;
  width: number;
  height: number;
  sizeBytes: number;
}}

export interface SurgvuCase {{
  caseId: string;
  dirName: string;
  parts: SurgvuPart[];
  totalDurationSeconds: number;
  /** Normalised tool classes this case's labels actually contain. */
  toolClasses: string[];
  taskClasses: string[];
  toolIntervalCount: number;
  taskIntervalCount: number;
}}

/** Directory the videos live in, relative to the surg workspace root. */
export const SURGVU_VIDEO_ROOT = 'surgvu24_videos_only';

export const SURGVU_CASES: SurgvuCase[] = {json.dumps(cases, indent=2)};

/** Every part across every case, flattened, in playback order. */
export const SURGVU_PARTS = SURGVU_CASES.flatMap((c) =>
  c.parts.map((p) => ({{ ...p, caseId: c.caseId, dirName: c.dirName }}))
);
"""


def emit_labels(tools: list[dict], tasks: list[dict]) -> str:
    return f"""{BANNER}
export interface SurgvuToolInterval {{
  case: string;
  part: number;
  start: number;
  end: number;
  /** snake_case class id, shared with the training pipeline. */
  label: string;
  display: string;
  /** Robot arm the tool was installed on: USM1..USM4. */
  arm: string;
  /** Manufacturer name from the release, e.g. "SureForm Stapler 60". */
  commercial: string;
}}

export interface SurgvuTaskInterval {{
  case: string;
  part: number;
  start: number;
  end: number;
  label: string;
  display: string;
}}

export const SURGVU_TOOL_INTERVALS: SurgvuToolInterval[] = {json.dumps(tools, indent=1)};

export const SURGVU_TASK_INTERVALS: SurgvuTaskInterval[] = {json.dumps(tasks, indent=1)};
"""


def main() -> int:
    if not VIDEO_ROOT.is_dir():
        print(f"Video root not found: {VIDEO_ROOT}")
        return 1
    if not LABEL_ROOT.is_dir():
        print(f"Label root not found: {LABEL_ROOT}")
        return 1

    print("Probing video parts...")
    cases = probe_parts()
    if not cases:
        print("No videos found.")
        return 1

    print("Resolving label intervals...")
    tools, tasks = build_intervals(cases)
    print(f"  {len(tools)} tool intervals, {len(tasks)} task intervals")

    out_dir = CHARLIE / "src" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "surgvuCases.ts").write_text(emit_cases(cases, tools, tasks), encoding="utf-8")
    (out_dir / "surgvuLabels.ts").write_text(emit_labels(tools, tasks), encoding="utf-8")

    print(f"Wrote {out_dir / 'surgvuCases.ts'}")
    print(f"Wrote {out_dir / 'surgvuLabels.ts'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
