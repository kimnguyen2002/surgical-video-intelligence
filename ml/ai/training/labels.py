"""
SurgVU label parsing.

Both label files describe **intervals**, not per-frame labels::

    install_case_part, install_case_time, uninstall_case_part,
    uninstall_case_time, arm, groundtruth_toolname   (tools.csv)
    ... , groundtruth_taskname                        (tasks.csv)

A row says "this tool was installed on this arm at (part P1, time T1) and
removed at (part P2, time T2)". Three consequences shape this module:

**Intervals can span video parts.** A tool installed near the end of part 3
and removed in part 5 covers all of part 4. Resolving that requires knowing how
long each part is, which comes from the frame manifest — so intervals are
resolved against the manifest rather than in isolation.

**Multiple arms carry tools simultaneously.** Tool presence is therefore
genuinely multi-label: the union of every arm's intervals covering a timestamp.

**Tool labels come from robot installation logs, not from vision.** A tool
counts as present while it is installed, including while it is off-screen or
occluded. This is weak supervision: a model that "wrongly" predicts absent for
an off-screen installed tool is penalised for being visually right. That is a
property of the dataset, not a bug in the pipeline, and it is why per-class
metrics here should be read as an upper bound on visual difficulty rather than
as clean detection accuracy.
"""

from __future__ import annotations

import csv
import logging
import re
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger("surgvu.labels")

# Column aliases — the released CSVs have varied slightly between versions.
_ALIASES: dict[str, tuple[str, ...]] = {
    "install_part": (
        "install_case_part", "install_part", "start_case_part", "start_part", "case_part",
    ),
    "install_time": (
        "install_case_time", "install_time", "start_case_time", "start_time",
    ),
    "uninstall_part": (
        "uninstall_case_part", "uninstall_part", "end_case_part", "stop_case_part", "stop_part",
    ),
    "uninstall_time": (
        "uninstall_case_time", "uninstall_time", "end_case_time", "stop_case_time", "stop_time",
    ),
    "arm": ("arm", "arm_id", "usm"),
    "tool": ("groundtruth_toolname", "tool_name", "toolname", "tool"),
    "task": ("groundtruth_taskname", "task_name", "taskname", "task", "step"),
}

#: Tool rows whose label normalises to one of these are not tool presence at
#: all. ``nan(camera in)`` marks the endoscope being (re)inserted — 1,449 rows
#: in the released v2 labels — and blank labels are padding, which
#: ``normalise_label`` folds to ``other``. Both must be dropped rather than kept,
#: or the camera becomes a phantom thirteenth instrument class.
#:
#: This applies to **tool** rows only. ``other`` is a legitimate *task* class
#: (it dominates ``tasks.csv``), so the task path must not use this list.
_DROP_TOOL_LABELS: frozenset[str] = frozenset(
    {"nancamera_in", "nancamera_out", "nan", "other", ""}
)

_CASE_PART_RE = re.compile(
    r"case[_\-]?(?P<case>\d+).*?(?:part|video)[_\-]?(?P<part>\d+)", re.IGNORECASE
)
_TRAILING_NUMBERS_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class LabelInterval:
    """One labelled interval, already resolved to a single video part."""

    case: str
    part: int
    start: float
    end: float
    label: str
    arm: str = ""

    def covers(self, timestamp: float) -> bool:
        return self.start <= timestamp < self.end


def normalise_label(raw: str) -> str:
    """
    Map a dataset label string to a snake_case class identifier.

    >>> normalise_label("Monopolar Curved Scissors")
    'monopolar_curved_scissors'
    >>> normalise_label("rectal artery/vein")
    'rectal_artery_vein'
    >>> normalise_label("permanent cautery hook/spatula")
    'permanent_cautery_hook_spatula'
    """
    text = (raw or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[/\\\-\s]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")

    # Canonical aliases for label spellings seen across dataset versions.
    aliases = {
        "cautery_hook": "permanent_cautery_hook_spatula",
        "permanent_cautery_hook": "permanent_cautery_hook_spatula",
        "cautery_spatula": "permanent_cautery_hook_spatula",
        # The cat1 COCO categories spell this singular; surgvu24 tools.csv
        # spells it plural. Same instrument — they must collapse to one class
        # or presence labels and boxes silently refer to different things.
        "monopolar_curved_scissor": "monopolar_curved_scissors",
        "tip_up_fenestrated_grasper_prograsp": "tip_up_fenestrated_grasper",
        "rectal_artery_vein_manipulation": "rectal_artery_vein",
        "rectal_artery_or_vein": "rectal_artery_vein",
        "general_skills_application": "skills_application",
        "retraction_and_collision_avoidance": "retraction_collision_avoidance",
        "tip_up_fenestrated_grasper_": "tip_up_fenestrated_grasper",
        "na": "other",
        "none": "other",
        "": "other",
    }
    return aliases.get(text, text)


def parse_case_part(value: str, default_case: str = "") -> tuple[str, int]:
    """
    Split a ``case_part`` cell into ``(case_id, part_index)``.

    The released v2 labels do **not** put a composite string in this column.
    ``tools.csv`` holds a bare float (``1.0``) and ``tasks.csv`` a bare int
    (``1``) — in both, the value is the *part number alone*, and the case is
    identified only by the containing directory. ``default_case`` therefore
    carries the authoritative case id, and a bare number is read as a part.

    Composite forms from other dataset releases are still accepted.

    >>> parse_case_part("case_042_video_part_003")
    ('042', 3)
    >>> parse_case_part("1.0", default_case="002")     # tools.csv
    ('002', 1)
    >>> parse_case_part("1", default_case="002")       # tasks.csv
    ('002', 1)
    """
    text = (value or "").strip()
    match = _CASE_PART_RE.search(text)
    if match:
        return match.group("case"), int(match.group("part"))

    # A bare number (``1``, ``1.0``, ``3.0``) is a part index, not a case id.
    # Reading it as a case is how every task label silently landed on part 0.
    try:
        return (default_case or "unknown"), int(float(text))
    except (TypeError, ValueError):
        pass

    numbers = _TRAILING_NUMBERS_RE.findall(text)
    if len(numbers) >= 2:
        return numbers[0], int(numbers[-1])
    if len(numbers) == 1:
        return (default_case or numbers[0]), int(numbers[0])
    return default_case or text or "unknown", 0


def parse_time(value: str) -> float:
    """
    Parse a timestamp cell into seconds.

    Accepts plain seconds, ``MM:SS``, and ``HH:MM:SS`` with optional fraction.

    >>> parse_time("125.5")
    125.5
    >>> parse_time("00:02:05")
    125.0
    """
    text = (value or "").strip()
    if not text:
        return 0.0
    if ":" in text:
        parts = text.split(":")
        try:
            numbers = [float(p) for p in parts]
        except ValueError:
            return 0.0
        seconds = 0.0
        for number in numbers:
            seconds = seconds * 60 + number
        return seconds
    try:
        return float(text)
    except ValueError:
        return 0.0


def _resolve_columns(fieldnames: Iterable[str]) -> dict[str, str]:
    """Map logical column names to the actual header names present."""
    lowered = {name.strip().lower(): name for name in fieldnames if name}
    resolved: dict[str, str] = {}
    for logical, candidates in _ALIASES.items():
        for candidate in candidates:
            if candidate in lowered:
                resolved[logical] = lowered[candidate]
                break
    return resolved


def case_id_from_path(csv_path: Path) -> str:
    """
    Recover the case id from a label file's location.

    Layout is ``labels/case_002/tools.csv``, and the directory is the only
    place the case id appears — the CSV itself never names it.

    >>> case_id_from_path(Path("labels/case_002/tools.csv"))
    '002'
    """
    name = Path(csv_path).parent.name
    numbers = _TRAILING_NUMBERS_RE.findall(name)
    return numbers[-1] if numbers else name


def load_label_rows(csv_path: Path, kind: str, case_id: str | None = None) -> list[dict]:
    """
    Read a SurgVU label CSV into raw, unresolved rows.

    ``kind`` is ``"tool"`` or ``"task"`` and selects which label column to use.
    ``case_id`` defaults to the one implied by the file's directory.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Label file not found: {csv_path}")

    case_id = case_id or case_id_from_path(csv_path)

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = _resolve_columns(reader.fieldnames or [])

        label_column = columns.get(kind)
        if label_column is None:
            raise ValueError(
                f"{csv_path.name} has no recognisable '{kind}' label column. "
                f"Found headers: {reader.fieldnames}"
            )
        if "install_part" not in columns or "install_time" not in columns:
            raise ValueError(
                f"{csv_path.name} is missing install_case_part/install_case_time. "
                f"Found headers: {reader.fieldnames}"
            )

        rows = []
        dropped = 0
        for row in reader:
            label = normalise_label(row.get(label_column, ""))
            if not label:
                continue
            if kind == "tool" and label in _DROP_TOOL_LABELS:
                dropped += 1
                continue
            install_case, install_part = parse_case_part(
                row.get(columns["install_part"], ""), case_id
            )
            install_time = parse_time(row.get(columns["install_time"], ""))

            if "uninstall_part" in columns and row.get(columns["uninstall_part"]):
                uninstall_case, uninstall_part = parse_case_part(
                    row[columns["uninstall_part"]], case_id
                )
                uninstall_time = parse_time(row.get(columns.get("uninstall_time", ""), ""))
            else:
                uninstall_case, uninstall_part = install_case, install_part
                uninstall_time = install_time

            rows.append(
                {
                    "case": install_case,
                    "start_part": install_part,
                    "start_time": install_time,
                    "end_case": uninstall_case,
                    "end_part": uninstall_part,
                    "end_time": uninstall_time,
                    "arm": row.get(columns.get("arm", ""), "") or "",
                    "label": label,
                }
            )
    if dropped:
        logger.debug(
            "%s: dropped %d non-instrument rows (camera in/out, blank)",
            csv_path.parent.name, dropped,
        )
    logger.info("Parsed %d rows from %s/%s", len(rows), csv_path.parent.name, csv_path.name)
    return rows


def resolve_intervals(
    rows: list[dict], part_durations: dict[tuple[str, int], float]
) -> list[LabelInterval]:
    """
    Expand raw rows into per-part intervals.

    ``part_durations`` maps ``(case, part)`` to that part's length in seconds
    (derived from the frame manifest). Rows spanning multiple parts are split
    across them; a part with no known duration is skipped with a warning
    rather than guessed at.
    """
    intervals: list[LabelInterval] = []
    unknown_parts: set[tuple[str, int]] = set()

    for row in rows:
        case = row["case"]
        start_part, end_part = row["start_part"], row["end_part"]

        if row["end_case"] != case:
            # Cross-case rows are malformed; keep only the opening part.
            end_part = start_part
            row = {**row, "end_time": part_durations.get((case, start_part), row["start_time"])}

        if end_part < start_part:
            start_part, end_part = end_part, start_part

        for part in range(start_part, end_part + 1):
            duration = part_durations.get((case, part))
            if duration is None:
                unknown_parts.add((case, part))
                continue
            start = row["start_time"] if part == start_part else 0.0
            end = row["end_time"] if part == end_part else duration
            start = max(0.0, min(start, duration))
            end = max(0.0, min(end, duration))
            if end <= start:
                continue
            intervals.append(
                LabelInterval(
                    case=case,
                    part=part,
                    start=start,
                    end=end,
                    label=row["label"],
                    arm=row["arm"],
                )
            )

    if unknown_parts:
        logger.warning(
            "%d (case, part) pairs referenced by labels have no extracted "
            "frames and were skipped, e.g. %s. Run extract_frames.py over the "
            "full video set to include them.",
            len(unknown_parts),
            sorted(unknown_parts)[:3],
        )
    logger.info("Resolved %d label intervals", len(intervals))
    return intervals


class IntervalIndex:
    """
    Fast timestamp → labels lookup.

    Per ``(case, part)``, interval boundaries are collapsed into a sorted list
    of change points with the active label set between each pair. Lookup is
    then a binary search, which matters when labelling millions of frames.
    """

    def __init__(self, intervals: Iterable[LabelInterval]):
        grouped: dict[tuple[str, int], list[LabelInterval]] = defaultdict(list)
        for interval in intervals:
            grouped[(interval.case, interval.part)].append(interval)

        self._boundaries: dict[tuple[str, int], list[float]] = {}
        self._labels: dict[tuple[str, int], list[frozenset[str]]] = {}

        for key, items in grouped.items():
            points = sorted({i.start for i in items} | {i.end for i in items})
            active: list[frozenset[str]] = []
            for left, right in zip(points, points[1:]):
                midpoint = (left + right) / 2
                active.append(
                    frozenset(i.label for i in items if i.start <= midpoint < i.end)
                )
            self._boundaries[key] = points
            self._labels[key] = active

    def labels_at(self, case: str, part: int, timestamp: float) -> frozenset[str]:
        key = (case, part)
        points = self._boundaries.get(key)
        if not points:
            return frozenset()
        index = bisect_right(points, timestamp) - 1
        if index < 0 or index >= len(self._labels[key]):
            return frozenset()
        return self._labels[key][index]

    def parts(self) -> list[tuple[str, int]]:
        return sorted(self._boundaries)

    def label_counts(self) -> dict[str, float]:
        """Total labelled seconds per class — used for class weighting."""
        totals: dict[str, float] = defaultdict(float)
        for key, points in self._boundaries.items():
            for (left, right), labels in zip(zip(points, points[1:]), self._labels[key]):
                for label in labels:
                    totals[label] += right - left
        return dict(totals)


def build_index(
    csv_path: Path,
    kind: str,
    part_durations: dict[tuple[str, int], float],
    case_id: str | None = None,
) -> IntervalIndex:
    """Convenience: parse a CSV and build a lookup index in one call."""
    rows = load_label_rows(csv_path, kind, case_id=case_id)
    return IntervalIndex(resolve_intervals(rows, part_durations))


def find_label_file(dataset_dir: Path, kind: str) -> Optional[Path]:
    """
    Locate ``tools.csv`` / ``tasks.csv`` anywhere under the dataset directory.

    The label archive nests them a few directories deep and the exact layout
    has changed between releases, so this searches rather than assuming.
    """
    dataset_dir = Path(dataset_dir)
    stem = "tools" if kind == "tool" else "tasks"
    exact = list(dataset_dir.rglob(f"{stem}.csv"))
    if exact:
        return exact[0]
    fuzzy = [p for p in dataset_dir.rglob("*.csv") if stem in p.stem.lower()]
    return fuzzy[0] if fuzzy else None
