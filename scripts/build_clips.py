"""
Choose and encode the demo clips.

The repository cannot carry the SurgVU videos — 8.9 GB for six cases, under a
data-use agreement that does not contemplate redistribution. But a demo whose
player is empty until you have downloaded 168 GB is not a demo. So one short
excerpt per case ships, chosen rather than taken from the start.

Why the choice matters
----------------------
The first ninety seconds of a robotic case is trocar placement: a still, dark
field with no instrument installed and no task labelled. An excerpt taken from
t=0 would show the detector finding nothing and the ground-truth panel saying
nothing, which reads as a broken app rather than an accurate one.

So each window is scored by how much *labelled* activity it contains — distinct
instruments, distinct tasks, and instrument changes within the window — and the
best-scoring window wins. The result is an excerpt where both the recorded
annotations and the models have something to say on nearly every frame.

Usage::

    python3 scripts/build_clips.py                  # encode
    python3 scripts/build_clips.py --dry-run        # just print the choices
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKSPACE = REPO.parent

LABELS_TS = REPO / "src" / "data" / "surgvuLabels.ts"
CASES_TS = REPO / "src" / "data" / "surgvuCases.ts"
OUT_DIR = REPO / "public" / "clips"

#: Where the real videos live. Overridable because only the author has them.
VIDEO_ROOTS = [
    WORKSPACE / "surgvu_videos",
    WORKSPACE / "surgvu24_videos_only",
]

#: Long enough to show a task boundary or an instrument change, short enough
#: that six of them stay inside a repository people will actually clone.
CLIP_SECONDS = 75

#: Candidate windows are tried on this grid. Finer than the clip length so a
#: window can straddle an interesting boundary rather than landing beside it.
STRIDE_SECONDS = 15

#: 960x540, CRF 23, 24 fps.
#:
#: The models consume 448px (detector) and 224px (classifiers), so by model
#: input alone 640px would be enough. It is not enough for the *viewer*: a
#: first pass at 640px/CRF 30 landed at 69 kbps, where diathermy smoke and the
#: specular highlights along an instrument shaft turn to mush. Someone
#: assessing this work is reading the video as well as the overlay, and 6 MB
#: of extra bandwidth is a cheaper price than footage that looks degraded.
TARGET_WIDTH = 960
CRF = 23
FPS = 24


def extract_array(source: str, name: str) -> list[dict]:
    """
    Pull one `export const NAME: T[] = [...]` array out of the generated TS.

    The opening bracket is found after the `=`, never by scanning forward from
    the name. The declaration is `export const SURGVU_CASES: SurgvuCase[] = [`,
    so the first `[` after the name belongs to the *type annotation* — matching
    on it returns the empty `[]` of `SurgvuCase[]` and every caller silently
    sees a dataset with nothing in it.

    Bracket depth is counted outside string literals only, so a label
    containing a bracket cannot end the array early.
    """
    marker = f"export const {name}"
    at = source.index(marker)
    start = source.index("[", source.index("=", at + len(marker)))

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(source)):
        ch = source[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return json.loads(source[start : i + 1])
    raise ValueError(f"unterminated array for {name}")


@dataclass
class Interval:
    case: str
    part: int
    start: float
    end: float
    label: str
    display: str


def load_intervals() -> tuple[list[Interval], list[Interval]]:
    source = LABELS_TS.read_text()
    tools = [Interval(r["case"], r["part"], r["start"], r["end"], r["label"], r["display"])
             for r in extract_array(source, "SURGVU_TOOL_INTERVALS")]
    tasks = [Interval(r["case"], r["part"], r["start"], r["end"], r["label"], r["display"])
             for r in extract_array(source, "SURGVU_TASK_INTERVALS")]
    return tools, tasks


def load_parts() -> list[dict]:
    source = CASES_TS.read_text()
    cases = extract_array(source, "SURGVU_CASES")
    parts = []
    for case in cases:
        for part in case["parts"]:
            parts.append({
                "caseId": case["caseId"],
                "dirName": case["dirName"],
                "part": part["part"],
                "filename": part["filename"],
                "durationSeconds": part["durationSeconds"],
            })
    return parts


def overlaps(iv: Interval, start: float, end: float) -> bool:
    return iv.end > start and iv.start < end


def score_window(
    start: float,
    end: float,
    tools: list[Interval],
    tasks: list[Interval],
) -> tuple[float, dict]:
    """
    How much a viewer would see happen in this window.

    Weights, and why:

    * **Distinct instruments (3x)** — the single most visible thing. A window
      with three instruments exercises the detector, the presence classifier
      and the overlay's colour coding at once.
    * **Distinct tasks (4x)** — rarer than instruments and harder to show. A
      window spanning two tasks demonstrates the task classifier actually
      changing its mind, which one spanning a single task cannot.
    * **Boundaries inside the window (2x)** — an install or uninstall that
      happens *on screen* is what makes the ground-truth track visibly move
      rather than sit still.
    * **Task coverage (6x, fractional)** — the fraction of the window with any
      task labelled. This is a multiplier on usefulness rather than a count:
      a window that is half unlabelled is half a demo.
    """
    in_tools = [t for t in tools if overlaps(t, start, end)]
    in_tasks = [t for t in tasks if overlaps(t, start, end)]

    distinct_tools = len({t.label for t in in_tools})
    distinct_tasks = len({t.label for t in in_tasks})

    boundaries = 0
    for iv in in_tools:
        if start < iv.start < end:
            boundaries += 1
        if start < iv.end < end:
            boundaries += 1

    covered = 0.0
    for iv in in_tasks:
        covered += min(end, iv.end) - max(start, iv.start)
    coverage = min(1.0, covered / (end - start))

    score = (
        3.0 * distinct_tools
        + 4.0 * distinct_tasks
        + 2.0 * min(boundaries, 4)
        + 6.0 * coverage
    )
    return score, {
        "distinctTools": distinct_tools,
        "distinctTasks": distinct_tasks,
        "boundaries": boundaries,
        "taskCoverage": round(coverage, 3),
        "tools": sorted({t.display for t in in_tools}),
        "tasks": sorted({t.display for t in in_tasks}),
    }


def pick_window(part: dict, tools: list[Interval], tasks: list[Interval]):
    part_tools = [t for t in tools if t.case == part["caseId"] and t.part == part["part"]]
    part_tasks = [t for t in tasks if t.case == part["caseId"] and t.part == part["part"]]
    duration = part["durationSeconds"]
    if duration < CLIP_SECONDS + 10:
        return None

    best = None
    start = 5.0
    while start + CLIP_SECONDS <= duration - 5:
        score, detail = score_window(start, start + CLIP_SECONDS, part_tools, part_tasks)
        if best is None or score > best[0]:
            best = (score, start, detail)
        start += STRIDE_SECONDS
    return best


def find_video(part: dict) -> Path | None:
    for root in VIDEO_ROOTS:
        candidate = root / part["dirName"] / part["filename"]
        if candidate.exists():
            return candidate
    return None


def encode(source: Path, start: float, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y",
            # -ss before -i seeks by keyframe index without decoding the
            # preceding hours. On a five-hour part, -ss after -i would decode
            # everything up to the cut point.
            "-ss", f"{start:.3f}",
            "-i", str(source),
            "-t", str(CLIP_SECONDS),
            "-vf", f"scale={TARGET_WIDTH}:-2,fps={FPS}",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", str(CRF),
            # yuv420p + faststart: without the first, Safari refuses the file;
            # without the second, playback waits for the whole download.
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seconds", type=int, default=CLIP_SECONDS)
    args = parser.parse_args()

    globals()["CLIP_SECONDS"] = args.seconds

    if not shutil.which("ffmpeg"):
        print("ffmpeg is not installed.", file=sys.stderr)
        return 1

    tools, tasks = load_intervals()
    parts = load_parts()

    # One clip per case: the best-scoring window across all of that case's parts.
    by_case: dict[str, tuple] = {}
    for part in parts:
        picked = pick_window(part, tools, tasks)
        if picked is None:
            continue
        score, start, detail = picked
        current = by_case.get(part["caseId"])
        if current is None or score > current[0]:
            by_case[part["caseId"]] = (score, start, detail, part)

    manifest = []
    for case_id in sorted(by_case):
        score, start, detail, part = by_case[case_id]
        source = find_video(part)
        name = f"case_{case_id}_excerpt.mp4"
        dest = OUT_DIR / name

        print(f"case_{case_id}  part {part['part']}  t={start:7.1f}s  score={score:5.1f}  "
              f"{detail['distinctTools']} tools, {detail['distinctTasks']} tasks, "
              f"coverage {detail['taskCoverage']:.0%}")
        print(f"    tools: {', '.join(detail['tools']) or '—'}")
        print(f"    tasks: {', '.join(detail['tasks']) or '—'}")

        if not args.dry_run:
            if source is None:
                print(f"    SKIPPED — source video not found for {part['filename']}")
                continue
            encode(source, start, dest)
            size = dest.stat().st_size
            print(f"    wrote {dest.relative_to(REPO)}  ({size / 1e6:.1f} MB)")

        manifest.append({
            "caseId": case_id,
            "dirName": part["dirName"],
            "part": part["part"],
            "sourceFilename": part["filename"],
            "file": f"clips/{name}",
            # The offset into the original part. Every ground-truth lookup adds
            # this to the player's currentTime, so the excerpt reads the same
            # annotations the full video would at that moment.
            "sourceStartSeconds": round(start, 3),
            "durationSeconds": CLIP_SECONDS,
            "width": TARGET_WIDTH,
            "fps": FPS,
            "selection": detail,
            "sizeBytes": dest.stat().st_size if dest.exists() else None,
        })

    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "clips.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Re-render the typed module the app imports. Kept in a separate script
        # so the manifest can be regenerated without re-encoding the video.
        from emit_clips_ts import render, CLIPS_TS

        CLIPS_TS.write_text(render(manifest))
        print(f"wrote {CLIPS_TS.relative_to(REPO)}")
        total = sum(m["sizeBytes"] or 0 for m in manifest)
        print(f"\n{len(manifest)} clips, {total / 1e6:.1f} MB total")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
