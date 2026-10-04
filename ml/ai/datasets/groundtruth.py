"""
Ground truth lookup for library videos.

The point of this module is epistemic. For a video that came from the SurgVU
release, the platform does not have to *guess* what instrument is in use at
t=1500s — the release says so, from the robot's own installation log. For a
cat1 clip it does not have to guess where the instrument is — there is a box.

So answers about library footage are grounded in recorded fact, and model
output is reserved for footage that has no annotation (an upload, a YouTube
retrieval, a live camera). Every record this module returns is stamped with its
:class:`Provenance`, and the surfaces that display it are expected to keep that
distinction visible. "The needle driver was installed at 00:39:45" and "a model
thinks it sees a needle driver" are different claims and must not be rendered
identically.

Durations come from the video files themselves; the label CSVs give intervals
in ``(part, seconds)`` and cannot be resolved to frames without knowing how
long each part runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ai.common import settings
from ai.datasets.coco import display_name as detection_display
from ai.datasets.corpora import VideoAsset, registry
from ai.training.config import TOOL_DISPLAY, TASK_DISPLAY
from ai.training.labels import IntervalIndex, build_index

logger = logging.getLogger("charlie.datasets.groundtruth")


class Provenance(str, Enum):
    """Where an assertion came from. Never inferred, always carried."""

    #: Recorded in the dataset release. Not a prediction.
    GROUND_TRUTH = "ground_truth"
    #: Produced by a model at inference time.
    PREDICTED = "predicted"
    #: No information available.
    NONE = "none"


@dataclass
class TimelineEntry:
    """One labelled interval on a video's timeline."""

    start: float
    end: float
    label: str
    display: str
    kind: str  # "tool" | "task" | "box"
    provenance: Provenance = Provenance.GROUND_TRUTH

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.end - self.start, 2),
            "label": self.label,
            "display": self.display,
            "kind": self.kind,
            "provenance": self.provenance.value,
        }


@dataclass
class FrameTruth:
    """Everything known for certain about one moment of one video."""

    asset_id: str
    timestamp: float
    tools: list[str] = field(default_factory=list)
    task: Optional[str] = None
    boxes: list[dict] = field(default_factory=list)
    provenance: Provenance = Provenance.NONE

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "timestamp": round(self.timestamp, 2),
            "tools": self.tools,
            "tools_display": [TOOL_DISPLAY.get(t, t.replace("_", " ")) for t in self.tools],
            "task": self.task,
            "task_display": TASK_DISPLAY.get(self.task or "", self.task or ""),
            "boxes": self.boxes,
            "provenance": self.provenance.value,
            "has_ground_truth": self.provenance is Provenance.GROUND_TRUTH,
        }


# ---------------------------------------------------------------------------
# Durations
# ---------------------------------------------------------------------------

_DURATION_CACHE = settings.cache_dir / "video_durations.json"


def _load_duration_cache() -> dict[str, float]:
    try:
        return json.loads(_DURATION_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_duration_cache(cache: dict[str, float]) -> None:
    try:
        _DURATION_CACHE.write_text(json.dumps(cache, indent=0), encoding="utf-8")
    except OSError as exc:  # pragma: no cover
        logger.debug("could not persist duration cache: %s", exc)


def probe_duration(path: Path) -> Optional[float]:
    """
    Duration in seconds, read from container metadata and cached on disk.

    Opening a 2 GB file just to read its header is fast, but doing it for 144
    files on every request is not — and the label pipeline needs every part's
    duration to resolve intervals that span parts.
    """
    path = Path(path)
    key = str(path.resolve())
    cache = _load_duration_cache()
    if key in cache:
        return cache[key] or None

    duration: Optional[float] = None
    try:
        import cv2  # noqa: PLC0415

        capture = cv2.VideoCapture(str(path))
        if capture.isOpened():
            frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0 and frames and frames > 0:
                duration = float(frames) / float(fps)
        capture.release()
    except Exception as exc:
        logger.debug("duration probe failed for %s: %s", path.name, exc)

    cache[key] = duration or 0.0
    _save_duration_cache(cache)
    return duration


@lru_cache(maxsize=256)
def part_durations_for_case(case: str) -> tuple[tuple[tuple[str, int], float], ...]:
    """
    ``{(case, part): seconds}`` for one surgvu24 case, as a hashable tuple.

    Returned as a tuple so it can live behind ``lru_cache``; callers convert
    to a dict.
    """
    record = registry.surgvu_cases.get(case)
    if record is None:
        return ()
    out = []
    for part, path in sorted(record.parts.items()):
        duration = probe_duration(path)
        if duration:
            out.append(((case, part), duration))
    return tuple(out)


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _surgvu_indices(case: str) -> tuple[Optional[IntervalIndex], Optional[IntervalIndex]]:
    """Tool and task interval indices for one case, or ``(None, None)``."""
    record = registry.surgvu_cases.get(case)
    if record is None or not record.has_labels:
        return None, None

    durations = dict(part_durations_for_case(case))
    if not durations:
        logger.debug("case %s has labels but no probeable video parts", case)
        return None, None

    tools = tasks = None
    try:
        if record.tools_csv:
            tools = build_index(record.tools_csv, "tool", durations, case_id=case)
    except Exception as exc:
        logger.error("case %s tools.csv failed: %s", case, exc)
    try:
        if record.tasks_csv:
            tasks = build_index(record.tasks_csv, "task", durations, case_id=case)
    except Exception as exc:
        logger.error("case %s tasks.csv failed: %s", case, exc)
    return tools, tasks


@lru_cache(maxsize=16)
def _cat1_frames(clip: str):
    from ai.datasets.coco import load_coco  # noqa: PLC0415

    entry = registry.cat1_videos.get(clip)
    if entry is None:
        return None
    try:
        return load_coco(entry.coco, clip, entry.video)
    except Exception as exc:
        logger.error("cat1 clip %s failed to load: %s", clip, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def truth_at(asset: VideoAsset, timestamp: float) -> FrameTruth:
    """What is known for certain at ``timestamp`` in ``asset``."""
    result = FrameTruth(asset_id=asset.asset_id, timestamp=timestamp)

    if asset.corpus == "surgvu24":
        tools_index, tasks_index = _surgvu_indices(asset.case)
        if tools_index is None and tasks_index is None:
            return result
        if tools_index is not None:
            result.tools = sorted(tools_index.labels_at(asset.case, asset.part, timestamp))
        if tasks_index is not None:
            active = sorted(tasks_index.labels_at(asset.case, asset.part, timestamp))
            result.task = active[0] if active else None
        result.provenance = Provenance.GROUND_TRUTH
        return result

    if asset.corpus == "cat1":
        clip = _cat1_frames(asset.case)
        if clip is None:
            return result
        # cat1 clips are 1 fps, so the frame index is the second.
        frame = clip.frames_by_index().get(int(timestamp))
        if frame is None:
            result.provenance = Provenance.GROUND_TRUTH  # annotated clip, empty second
            return result
        result.boxes = [
            {
                "class": box.class_name,
                "display": detection_display(box.class_name),
                "x": box.x, "y": box.y, "w": box.w, "h": box.h,
                "frame_width": frame.width, "frame_height": frame.height,
            }
            for box in frame.boxes
        ]
        result.tools = sorted(frame.present_classes)
        result.provenance = Provenance.GROUND_TRUTH
        return result

    return result


def timeline_for(asset: VideoAsset) -> list[TimelineEntry]:
    """
    The video's full annotated timeline, merged and sorted.

    This is what makes a library video immediately useful: the scrubber is
    populated with real events before any model has run.
    """
    entries: list[TimelineEntry] = []

    if asset.corpus == "surgvu24":
        tools_index, tasks_index = _surgvu_indices(asset.case)
        for index, kind, display_map in (
            (tools_index, "tool", TOOL_DISPLAY),
            (tasks_index, "task", TASK_DISPLAY),
        ):
            if index is None:
                continue
            entries.extend(_entries_from_index(index, asset, kind, display_map))

    elif asset.corpus == "cat1":
        clip = _cat1_frames(asset.case)
        if clip is not None:
            entries.extend(_entries_from_cat1(clip))

    entries.sort(key=lambda e: (e.start, e.kind, e.label))
    return entries


def _entries_from_index(
    index: IntervalIndex, asset: VideoAsset, kind: str, display_map: dict
) -> list[TimelineEntry]:
    """
    Recover contiguous runs of each label from the interval index.

    The index stores change points and the active label set between them, so a
    label that persists across several change points must be re-joined —
    otherwise the timeline shows one instrument as a dozen adjacent slivers.
    """
    key = (asset.case, asset.part)
    boundaries = index._boundaries.get(key)  # noqa: SLF001 - same package
    actives = index._labels.get(key)  # noqa: SLF001
    if not boundaries or not actives:
        return []

    open_runs: dict[str, float] = {}
    out: list[TimelineEntry] = []

    for i, labels in enumerate(actives):
        start, end = boundaries[i], boundaries[i + 1]
        for label in labels:
            open_runs.setdefault(label, start)
        for label in list(open_runs):
            if label not in labels:
                out.append(
                    TimelineEntry(
                        start=open_runs.pop(label), end=start, label=label,
                        display=display_map.get(label, label.replace("_", " ")), kind=kind,
                    )
                )
    tail = boundaries[-1]
    for label, start in open_runs.items():
        out.append(
            TimelineEntry(
                start=start, end=tail, label=label,
                display=display_map.get(label, label.replace("_", " ")), kind=kind,
            )
        )
    return out


def _entries_from_cat1(clip) -> list[TimelineEntry]:
    """Collapse per-frame cat1 boxes into per-class presence runs."""
    frames = sorted(clip.frames, key=lambda f: f.index)
    open_runs: dict[str, int] = {}
    out: list[TimelineEntry] = []
    previous = -1

    for frame in frames:
        present = frame.present_classes
        # A gap in annotated frames closes every run: absence of annotation is
        # not evidence of absence of instrument.
        gap = previous >= 0 and frame.index != previous + 1
        for label in list(open_runs):
            if gap or label not in present:
                out.append(
                    TimelineEntry(
                        start=float(open_runs.pop(label)), end=float(frame.index),
                        label=label, display=detection_display(label), kind="box",
                    )
                )
        for label in present:
            open_runs.setdefault(label, frame.index)
        previous = frame.index

    for label, start in open_runs.items():
        out.append(
            TimelineEntry(
                start=float(start), end=float(previous + 1), label=label,
                display=detection_display(label), kind="box",
            )
        )
    return out


def summary_for(asset: VideoAsset) -> dict:
    """
    A compact, factual description of a video — the assistant's grounding.

    Everything here is read from the release's own annotations, so an answer
    built on it can be stated as fact rather than hedged as a prediction.
    """
    entries = timeline_for(asset)
    duration = probe_duration(asset.path)

    tools: dict[str, float] = {}
    tasks: dict[str, float] = {}
    for entry in entries:
        bucket = tools if entry.kind in {"tool", "box"} else tasks
        bucket[entry.display] = bucket.get(entry.display, 0.0) + (entry.end - entry.start)

    return {
        "asset_id": asset.asset_id,
        "title": asset.title,
        "corpus": asset.corpus,
        "case": asset.case,
        "part": asset.part,
        "duration_s": round(duration, 1) if duration else None,
        "has_ground_truth": bool(entries),
        "provenance": (
            Provenance.GROUND_TRUTH.value if entries else Provenance.NONE.value
        ),
        "events": len(entries),
        "instruments": [
            {"display": name, "seconds": round(seconds, 1)}
            for name, seconds in sorted(tools.items(), key=lambda kv: -kv[1])
        ],
        "tasks": [
            {"display": name, "seconds": round(seconds, 1)}
            for name, seconds in sorted(tasks.items(), key=lambda kv: -kv[1])
        ],
    }


def clear_caches() -> None:
    _surgvu_indices.cache_clear()
    _cat1_frames.cache_clear()
    part_durations_for_case.cache_clear()
