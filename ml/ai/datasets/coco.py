"""
cat1 COCO bounding-box annotations.

This is the only part of the SurgVU release that carries **spatial** ground
truth. The 155 surgvu24 cases give tool *presence* derived from robot
installation logs; cat1 gives 14-class boxes on decoded 1 fps frames. Anything
that claims a box on screen is either trained or evaluated here.

Two things about this data drive the design:

**It is a test set.** These are the public cat1 annotations, released for
scoring. Training a detector on them is legitimate as a demonstrator but it is
not a held-out result, so :func:`split_clips` partitions by *clip* and every
caller is expected to report which clips it trained on. A frame-level split
would put frame *n* in train and frame *n+1* in val — consecutive 1 fps frames
of the same instrument in the same operation — and report a number that means
nothing.

**It is positives-only, and densely covered.** The clips ship already decoded
at exactly 1 fps, so a COCO ``image_id`` *is* the frame index *is* the second
offset — no manifest or timestamp arithmetic is needed anywhere. 5,178 of the
5,240 frames across the seven clips are annotated (98.8%), and every annotated
frame carries at least one box. There are therefore **no empty-scene negatives
in this data at all**. A detector trained on it learns background only from the
non-box regions of frames that do contain an instrument, and will be biased
toward always predicting something. That is a property of the annotation set,
not of the model, and :func:`dataset_report` surfaces it so it cannot be
mistaken for a training bug.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ai.training.labels import normalise_label

logger = logging.getLogger("charlie.datasets.coco")


#: The cat1 category vocabulary, in COCO id order, normalised to the same
#: snake_case identifiers the presence pipeline uses. Twelve of these coincide
#: with ``ai.training.config.TOOL_CLASSES``; ``bipolar_dissector`` and
#: ``suction_irrigator`` are additional — they appear in surgvu24 ``tools.csv``
#: too (6 and 23 rows) but were left out of the 12-class presence vocabulary.
#:
#: Index in this list is the detection class id. It is pinned to the COCO
#: ``category_id`` order in the released files and must not be re-sorted:
#: exported YOLO labels reference it by position.
DETECTION_CLASSES: list[str] = [
    "grasping_retractor",              # 0
    "cadiere_forceps",                 # 1
    "bipolar_forceps",                 # 2
    "force_bipolar",                   # 3
    "clip_applier",                    # 4
    "stapler",                         # 5
    "permanent_cautery_hook_spatula",  # 6
    "monopolar_curved_scissors",       # 7
    "vessel_sealer",                   # 8
    "tip_up_fenestrated_grasper",      # 9
    "bipolar_dissector",               # 10
    "needle_driver",                   # 11
    "prograsp_forceps",                # 12
    "suction_irrigator",               # 13
]

DETECTION_DISPLAY: dict[str, str] = {
    "grasping_retractor": "Grasping retractor",
    "cadiere_forceps": "Cadiere forceps",
    "bipolar_forceps": "Bipolar forceps",
    "force_bipolar": "Force bipolar",
    "clip_applier": "Clip applier",
    "stapler": "Stapler",
    "permanent_cautery_hook_spatula": "Permanent cautery hook/spatula",
    "monopolar_curved_scissors": "Monopolar curved scissors",
    "vessel_sealer": "Vessel sealer",
    "tip_up_fenestrated_grasper": "Tip-up fenestrated grasper",
    "bipolar_dissector": "Bipolar dissector",
    "needle_driver": "Needle driver",
    "prograsp_forceps": "Prograsp forceps",
    "suction_irrigator": "Suction irrigator",
}


@dataclass(frozen=True)
class Box:
    """One annotation. ``xywh`` is COCO's top-left origin, in pixels."""

    class_id: int
    x: float
    y: float
    w: float
    h: float

    @property
    def class_name(self) -> str:
        return DETECTION_CLASSES[self.class_id]

    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def to_yolo(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Centre-x, centre-y, w, h — all normalised to [0, 1]."""
        return (
            _clamp01((self.x + self.w / 2) / width),
            _clamp01((self.y + self.h / 2) / height),
            _clamp01(self.w / width),
            _clamp01(self.h / height),
        )


@dataclass
class Frame:
    """One decoded frame and every box on it."""

    clip: str
    #: Index into the clip's 1 fps frame sequence — also the second offset,
    #: since these clips are decoded at exactly 1 fps.
    index: int
    width: int
    height: int
    file_name: str
    boxes: list[Box] = field(default_factory=list)

    @property
    def timestamp_s(self) -> float:
        return float(self.index)

    @property
    def is_background(self) -> bool:
        return not self.boxes

    @property
    def present_classes(self) -> frozenset[str]:
        return frozenset(b.class_name for b in self.boxes)


@dataclass
class ClipAnnotations:
    """Every annotated frame of one cat1 clip, in temporal order."""

    clip: str
    video: Path
    frames: list[Frame]

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def fps(self) -> float:
        """cat1 clips are released pre-decoded at 1 fps. Index == second."""
        return 1.0

    @property
    def boxed_frames(self) -> list[Frame]:
        return [f for f in self.frames if f.boxes]

    def class_counts(self) -> Counter:
        counts: Counter = Counter()
        for frame in self.frames:
            for box in frame.boxes:
                counts[box.class_name] += 1
        return counts

    def frames_by_index(self) -> dict[int, Frame]:
        return {f.index: f for f in self.frames}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _category_map(categories: Sequence[dict]) -> dict[int, int]:
    """
    Map the file's ``category_id`` values onto :data:`DETECTION_CLASSES`.

    Resolved by *name*, never by position. The released files happen to list
    categories in id order, but relying on that would silently relabel every
    box if a future release reorders them — turning a needle driver into a
    stapler with no error anywhere.
    """
    index = {name: i for i, name in enumerate(DETECTION_CLASSES)}
    mapping: dict[int, int] = {}
    unknown: list[str] = []
    for entry in categories:
        raw = str(entry.get("name", ""))
        normalised = normalise_label(raw)
        if normalised in index:
            mapping[int(entry["id"])] = index[normalised]
        else:
            unknown.append(raw)
    if unknown:
        logger.warning(
            "COCO categories not in DETECTION_CLASSES and therefore dropped: %s",
            sorted(set(unknown)),
        )
    return mapping


def load_coco(coco_path: Path, clip: str, video: Path) -> ClipAnnotations:
    """Read one ``*_coco.json`` into :class:`ClipAnnotations`."""
    coco_path = Path(coco_path)
    payload = json.loads(coco_path.read_text(encoding="utf-8"))

    cat_map = _category_map(payload.get("categories", []))

    by_image: dict[int, list[Box]] = defaultdict(list)
    skipped = 0
    for ann in payload.get("annotations", []):
        class_id = cat_map.get(int(ann.get("category_id", -1)))
        if class_id is None:
            skipped += 1
            continue
        bbox = ann.get("bbox") or []
        if len(bbox) != 4:
            skipped += 1
            continue
        x, y, w, h = (float(v) for v in bbox)
        if w <= 0 or h <= 0:
            skipped += 1
            continue
        by_image[int(ann["image_id"])].append(Box(class_id, x, y, w, h))

    frames: list[Frame] = []
    for image in payload.get("images", []):
        image_id = int(image["id"])
        frames.append(
            Frame(
                clip=clip,
                # ``id`` is the 1 fps frame index; ``file_name`` is the same
                # number zero-padded. Prefer the id — it is an int already.
                index=image_id,
                width=int(image.get("width", 0)),
                height=int(image.get("height", 0)),
                file_name=str(image.get("file_name", f"{image_id:010d}.jpg")),
                boxes=by_image.get(image_id, []),
            )
        )
    frames.sort(key=lambda f: f.index)

    if skipped:
        logger.info("%s: skipped %d unusable annotations", coco_path.name, skipped)
    return ClipAnnotations(clip=clip, video=Path(video), frames=frames)


def load_all(registry=None) -> dict[str, ClipAnnotations]:
    """Load every discovered cat1 clip."""
    if registry is None:
        from ai.datasets.corpora import registry as default_registry

        registry = default_registry

    out: dict[str, ClipAnnotations] = {}
    for name, clip in registry.cat1_videos.items():
        try:
            out[name] = load_coco(clip.coco, name, clip.video)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.error("cat1 clip %s failed to load: %s", name, exc)
    return out


def split_clips(
    clips: Iterable[str],
    val_fraction: float = 0.3,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Partition clip names into train and validation, **by clip**.

    With only seven clips this is a coarse split, and that is the honest
    position: it is what the data supports. Sorting before shuffling keeps the
    partition reproducible regardless of filesystem ordering.
    """
    names = sorted(clips, key=lambda c: (len(c), c))
    if len(names) < 2:
        return names, []

    rng = random.Random(seed)
    shuffled = names[:]
    rng.shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * val_fraction))
    n_val = min(n_val, len(shuffled) - 1)  # never leave train empty
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


def dataset_report(clips: dict[str, ClipAnnotations]) -> dict:
    """Counts worth printing before a training run commits hours to this data."""
    total_frames = sum(len(c) for c in clips.values())
    boxed = sum(len(c.boxed_frames) for c in clips.values())
    per_class: Counter = Counter()
    for clip in clips.values():
        per_class.update(clip.class_counts())

    return {
        "clips": len(clips),
        "frames": total_frames,
        "frames_with_boxes": boxed,
        "frames_background": total_frames - boxed,
        "boxes": int(sum(per_class.values())),
        "classes_present": len(per_class),
        "classes_absent": [c for c in DETECTION_CLASSES if c not in per_class],
        "per_class": dict(per_class.most_common()),
        "per_clip": {
            name: {"frames": len(c), "boxes": int(sum(c.class_counts().values()))}
            for name, c in sorted(clips.items())
        },
    }


def display_name(class_name: str) -> str:
    return DETECTION_DISPLAY.get(class_name, class_name.replace("_", " ").capitalize())
