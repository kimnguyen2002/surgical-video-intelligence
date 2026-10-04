"""
PyTorch datasets for SurgVU.

``SurgVUToolDataset``
    Per-frame multi-label tool presence. One sample = one frame, target = a
    12-dim binary vector.

``SurgVUTaskDataset``
    Clip-level surgical task recognition. One sample = ``clip_length`` frames
    sampled with ``frame_step``, target = the majority task label over the
    clip.

Both are built from the extraction manifest plus the interval index, and both
split **by case**. Frames within one operation are extremely correlated — the
same patient, lighting, camera, and instruments — so a random frame-level split
would put near-duplicates of validation frames into training and report
accuracy that collapses on genuinely unseen cases.

Augmentation is surgical-specific: horizontal flips and mild colour jitter are
safe, but vertical flips and rotations are not — endoscopic video has a
consistent gravity and horizon, and a model trained on upside-down frames
wastes capacity learning an orientation that never occurs.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ai.training.config import config
from ai.training.extract_frames import load_manifest, part_durations
from ai.training.labels import IntervalIndex, build_index, find_label_file

logger = logging.getLogger("surgvu.dataset")

try:  # Torch is optional at import time so `--help` works without it.
    import torch
    from torch.utils.data import Dataset

    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False

    class Dataset:  # type: ignore[no-redef]
        pass


@dataclass
class FrameRecord:
    path: str
    case: str
    part: int
    timestamp: float


# ---------------------------------------------------------------------------
# Shared assembly
# ---------------------------------------------------------------------------
def build_frame_records(manifest: dict) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for entry in manifest["parts"]:
        directory = Path(entry["dir"])
        for frame in entry["frames"]:
            records.append(
                FrameRecord(
                    path=str(directory / frame["file"]),
                    case=entry["case"],
                    part=entry["part"],
                    timestamp=frame["timestamp"],
                )
            )
    return records


def split_cases(
    cases: list[str],
    val_fraction: float = None,
    test_fraction: float = None,
    seed: int = None,
) -> dict[str, set[str]]:
    """Deterministic case-level split."""
    val_fraction = config.val_fraction if val_fraction is None else val_fraction
    test_fraction = config.test_fraction if test_fraction is None else test_fraction
    seed = config.split_seed if seed is None else seed

    ordered = sorted(set(cases))
    rng = random.Random(seed)
    rng.shuffle(ordered)

    total = len(ordered)
    n_test = max(1, int(total * test_fraction)) if total > 2 else 0
    n_val = max(1, int(total * val_fraction)) if total > 1 else 0

    test = set(ordered[:n_test])
    val = set(ordered[n_test : n_test + n_val])
    train = set(ordered[n_test + n_val :])

    if not train:  # Tiny datasets (smoke tests) — keep training non-empty.
        train, val, test = set(ordered), set(), set()

    logger.info(
        "Case split — train %d / val %d / test %d", len(train), len(val), len(test)
    )
    return {"train": train, "val": val, "test": test}


def default_transform(train: bool, image_size: Optional[int] = None):
    """Surgical-appropriate augmentation pipeline."""
    import torchvision.transforms as T

    size = image_size or config.image_size
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean=config.mean, std=config.std),
                # Simulates smoke, specular glare, and occlusion by tissue.
                T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
            ]
        )
    return T.Compose(
        [
            T.Resize(int(size * 1.14)),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=config.mean, std=config.std),
        ]
    )


# ---------------------------------------------------------------------------
# Tool detection
# ---------------------------------------------------------------------------
class SurgVUToolDataset(Dataset):
    """Multi-label tool presence, one frame per sample."""

    def __init__(
        self,
        frames_dir: Optional[str] = None,
        dataset_dir: Optional[str] = None,
        split: str = "train",
        classes: Optional[list[str]] = None,
        transform=None,
        drop_unlabelled: bool = True,
    ):
        if not _TORCH:
            raise ImportError("PyTorch is required. Install: pip install torch torchvision")

        self.classes = classes or config.tool_classes
        self.class_to_index = {name: i for i, name in enumerate(self.classes)}
        self.split = split
        self.transform = transform if transform is not None else default_transform(split == "train")

        frames_root = Path(frames_dir or config.frames_dir)
        dataset_root = Path(dataset_dir or config.dataset_dir)

        manifest = load_manifest(frames_root)
        label_file = find_label_file(dataset_root, "tool")
        if label_file is None:
            raise FileNotFoundError(
                f"No tools.csv found under {dataset_root}. Download the labels archive:\n"
                "  python -m ai.training.download_dataset --dest data/surgvu --only labels --extract"
            )
        self.index: IntervalIndex = build_index(label_file, "tool", part_durations(manifest))

        splits = split_cases(manifest["cases"])
        wanted = splits[split]

        records = [r for r in build_frame_records(manifest) if r.case in wanted]
        self.samples: list[tuple[FrameRecord, list[float]]] = []
        skipped = 0
        for record in records:
            labels = self.index.labels_at(record.case, record.part, record.timestamp)
            known = [l for l in labels if l in self.class_to_index]
            if drop_unlabelled and not known:
                # No tool installed on any arm: real, but it dominates the
                # dataset and teaches the model to predict all-zeros.
                skipped += 1
                continue
            target = [0.0] * len(self.classes)
            for label in known:
                target[self.class_to_index[label]] = 1.0
            self.samples.append((record, target))

        logger.info(
            "SurgVUToolDataset[%s]: %d frames (%d unlabelled skipped)",
            split,
            len(self.samples),
            skipped,
        )
        if not self.samples:
            logger.warning(
                "Split '%s' is empty — check that label case ids match video "
                "filenames.",
                split,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        record, target = self.samples[index]
        try:
            image = Image.open(record.path).convert("RGB")
        except (OSError, ValueError) as exc:
            # One unreadable JPEG must not kill an epoch; substitute a
            # neighbour so the batch keeps its shape.
            logger.warning("Unreadable frame %s (%s)", record.path, exc)
            return self[(index + 1) % len(self)]

        tensor = self.transform(image)
        return tensor, torch.tensor(target, dtype=torch.float32)

    # -- statistics --------------------------------------------------------
    def class_counts(self) -> list[int]:
        counts = [0] * len(self.classes)
        for _, target in self.samples:
            for i, value in enumerate(target):
                counts[i] += int(value)
        return counts

    def pos_weight(self) -> "torch.Tensor":
        """
        ``pos_weight`` for ``BCEWithLogitsLoss``: negatives/positives per class.

        SurgVU tool frequency is severely imbalanced — needle drivers appear
        constantly, staplers rarely. Without this the loss is minimised by
        predicting "absent" for rare tools, which scores well on accuracy and
        is useless in practice. Weights are capped so a class with a handful of
        examples cannot destabilise training.
        """
        counts = self.class_counts()
        total = len(self.samples)
        weights = []
        for count in counts:
            if count == 0:
                weights.append(1.0)
            else:
                weights.append(min((total - count) / count, 50.0))
        return torch.tensor(weights, dtype=torch.float32)

    def describe(self) -> dict:
        counts = self.class_counts()
        return {
            "split": self.split,
            "frames": len(self.samples),
            "classes": {
                name: {
                    "count": count,
                    "prevalence": round(count / max(len(self.samples), 1), 4),
                }
                for name, count in zip(self.classes, counts)
            },
        }


# ---------------------------------------------------------------------------
# Step / task recognition
# ---------------------------------------------------------------------------
class SurgVUTaskDataset(Dataset):
    """Clip-level surgical task recognition."""

    def __init__(
        self,
        frames_dir: Optional[str] = None,
        dataset_dir: Optional[str] = None,
        split: str = "train",
        classes: Optional[list[str]] = None,
        clip_length: Optional[int] = None,
        frame_step: Optional[int] = None,
        clip_stride: Optional[int] = None,
        transform=None,
    ):
        if not _TORCH:
            raise ImportError("PyTorch is required. Install: pip install torch torchvision")

        self.classes = classes or config.task_classes
        self.class_to_index = {name: i for i, name in enumerate(self.classes)}
        self.clip_length = clip_length or config.clip_length
        self.frame_step = frame_step or config.frame_step
        self.clip_stride = clip_stride or config.clip_stride
        self.split = split
        self.transform = transform if transform is not None else default_transform(split == "train")

        frames_root = Path(frames_dir or config.frames_dir)
        dataset_root = Path(dataset_dir or config.dataset_dir)

        manifest = load_manifest(frames_root)
        label_file = find_label_file(dataset_root, "task")
        if label_file is None:
            raise FileNotFoundError(
                f"No tasks.csv found under {dataset_root}. Download the labels archive:\n"
                "  python -m ai.training.download_dataset --dest data/surgvu --only labels --extract"
            )
        self.index = build_index(label_file, "task", part_durations(manifest))

        splits = split_cases(manifest["cases"])
        wanted = splits[split]

        # Clips are built within a single part so they never jump across a cut.
        self.clips: list[tuple[list[FrameRecord], int]] = []
        span = self.clip_length * self.frame_step

        for entry in manifest["parts"]:
            if entry["case"] not in wanted:
                continue
            directory = Path(entry["dir"])
            frames = [
                FrameRecord(
                    path=str(directory / f["file"]),
                    case=entry["case"],
                    part=entry["part"],
                    timestamp=f["timestamp"],
                )
                for f in entry["frames"]
            ]
            for start in range(0, max(len(frames) - span + 1, 0), self.clip_stride):
                window = frames[start : start + span : self.frame_step]
                if len(window) < self.clip_length:
                    continue
                labels = [
                    next(iter(self.index.labels_at(f.case, f.part, f.timestamp)), None)
                    for f in window
                ]
                labels = [l for l in labels if l in self.class_to_index]
                if not labels:
                    continue
                # Majority label over the clip; a clip straddling a boundary
                # takes the task it mostly belongs to.
                majority = Counter(labels).most_common(1)[0][0]
                self.clips.append((window, self.class_to_index[majority]))

        logger.info("SurgVUTaskDataset[%s]: %d clips", split, len(self.clips))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int):
        from PIL import Image

        window, label = self.clips[index]
        tensors = []
        for record in window:
            try:
                image = Image.open(record.path).convert("RGB")
            except (OSError, ValueError):
                image = Image.new("RGB", (config.image_size, config.image_size))
            tensors.append(self.transform(image))
        # (T, C, H, W) — the temporal model permutes as its backbone needs.
        clip = torch.stack(tensors)
        return clip, torch.tensor(label, dtype=torch.long)

    def class_counts(self) -> list[int]:
        counts = [0] * len(self.classes)
        for _, label in self.clips:
            counts[label] += 1
        return counts

    def class_weights(self) -> "torch.Tensor":
        """Inverse-frequency weights for cross-entropy."""
        counts = self.class_counts()
        total = sum(counts) or 1
        weights = [
            (total / (len(counts) * count)) if count else 1.0 for count in counts
        ]
        return torch.tensor(weights, dtype=torch.float32)

    def describe(self) -> dict:
        counts = self.class_counts()
        return {
            "split": self.split,
            "clips": len(self.clips),
            "clip_length": self.clip_length,
            "classes": dict(zip(self.classes, counts)),
        }
