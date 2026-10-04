"""
Instrument-presence classifier trained on the exported cat1 frames.

Run from the repository root::

    python -m ai.training.export_detection --train-stride 2   # once
    python -m ai.training.train_presence --epochs 20
    python -m ai.training.train_presence --smoke-test

Why this exists
---------------
Grad-CAM needs a **classifier**, not a detector. It works by backpropagating a
single class logit to a convolutional feature map; a YOLO detection head has no
such logit to attribute. So although the platform has a trained detector, the
whole explainability surface stayed dark — ``FrameAnalyzer`` reports
``predictions_available: false`` and returns before it ever reaches the
Grad-CAM branch.

The other route to a classifier is `train_tool_detection`, which learns from
the 155 surgvu24 cases. That is the better model, but it needs frame extraction
over ~168 GB of video first — tens of hours on a CPU-only machine.

This module takes the shortcut that is already paid for: the cat1 export has
5,178 decoded frames sitting on disk with per-frame boxes, and a box implies
presence. Collapsing boxes to a multi-label presence vector gives a genuine
training target at zero extra decode cost, and produces a checkpoint in exactly
the format `FrameAnalyzer` loads.

What the resulting model is
---------------------------
Trained on five clips, validated on two held-out clips, over the 8 of 14
classes that actually occur in cat1. It is a demonstrator that makes the
explainability path real and inspectable — not a validated classifier. It is
saved with its class list and provenance so every consumer can say what it is.

Unlike the surgvu24 presence labels, these targets come from **vision**: a
class is positive only when an annotator drew a box on that frame. So this
model has no off-screen-instrument problem — a useful contrast with
`train_tool_detection`, and the reason its Grad-CAM maps are worth looking at.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

from ai.datasets.coco import DETECTION_CLASSES
from ai.training.config import REPO_ROOT, config

logger = logging.getLogger("surgvu.train_presence")

DEFAULT_DATA = REPO_ROOT / "data" / "cat1_yolo"
DEFAULT_OUT = REPO_ROOT / "checkpoints"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Cat1PresenceDataset:
    """
    Multi-label presence over the exported cat1 frames.

    Reads the YOLO label files rather than the COCO JSON so that the split
    already materialised on disk by ``export_detection`` is reused exactly —
    two different notions of "the training split" is a bug waiting to happen.
    """

    def __init__(self, root: Path, split: str, train: bool, image_size: int):
        from ai.training.dataset import default_transform

        self.images_dir = Path(root) / "images" / split
        self.labels_dir = Path(root) / "labels" / split
        self.transform = default_transform(train=train, image_size=image_size)
        self.num_classes = len(DETECTION_CLASSES)

        if not self.images_dir.is_dir():
            raise SystemExit(
                f"No exported frames at {self.images_dir}.\n"
                "Build them first:  python -m ai.training.export_detection"
            )

        self.samples: list[tuple[Path, list[int]]] = []
        for image_path in sorted(self.images_dir.glob("*.jpg")):
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            present = [0] * self.num_classes
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        class_id = int(parts[0])
                    except ValueError:
                        continue
                    if 0 <= class_id < self.num_classes:
                        present[class_id] = 1
            self.samples.append((image_path, present))

        if not self.samples:
            raise SystemExit(f"No frames found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image

        path, present = self.samples[index]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        return self.transform(image), torch.tensor(present, dtype=torch.float32)

    def positive_counts(self) -> list[int]:
        counts = [0] * self.num_classes
        for _, present in self.samples:
            for i, value in enumerate(present):
                counts[i] += value
        return counts


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _pos_weight(counts: list[int], total: int, cap: float = 20.0):
    """
    Per-class positive weighting for the BCE loss.

    Instrument prevalence in cat1 spans two orders of magnitude — needle
    drivers appear in thousands of frames, vessel sealers in thirteen.
    Unweighted BCE is minimised by predicting "absent" for the rare classes.
    The cap stops a class with a handful of examples from dominating the
    gradient and destabilising training.

    Classes that never occur get weight 1.0: there is no signal to weight, and
    a large weight on an all-negative class is pure noise.
    """
    import torch

    weights = []
    for positive in counts:
        if positive <= 0:
            weights.append(1.0)
            continue
        negative = max(1, total - positive)
        weights.append(min(cap, negative / positive))
    return torch.tensor(weights, dtype=torch.float32)


def train(
    data_dir: Path = DEFAULT_DATA,
    *,
    model_name: str = "convnextv2_atto.fcmae_ft_in1k",
    epochs: int = 20,
    batch_size: int = 16,
    image_size: int = 224,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.05,
    workers: int = 0,
    device: str = "auto",
    out_dir: Path = DEFAULT_OUT,
    threads: Optional[int] = None,
    patience: int = 6,
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from ai.training.metrics import mean_average_precision
    from ai.training.models import SurgVUToolModel, save_checkpoint
    from ai.training.train_detection import configure_cpu_threads

    resolved = (
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    if resolved == "cpu":
        configure_cpu_threads(threads)

    train_set = Cat1PresenceDataset(data_dir, "train", train=True, image_size=image_size)
    val_set = Cat1PresenceDataset(data_dir, "val", train=False, image_size=image_size)

    counts = train_set.positive_counts()
    present_classes = [
        DETECTION_CLASSES[i] for i, c in enumerate(counts) if c > 0
    ]
    logger.info("train %d frames · val %d frames", len(train_set), len(val_set))
    logger.info("classes with positives: %d/%d", len(present_classes), len(DETECTION_CLASSES))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=workers
    )

    model = SurgVUToolModel(
        model_name=model_name,
        num_classes=len(DETECTION_CLASSES),
        pretrained=True,
    ).to(resolved)

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=_pos_weight(counts, len(train_set)).to(resolved)
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = min(steps_per_epoch, total_steps // 10)

    def lr_at(step: int) -> float:
        # Warmup then cosine decay. Fine-tuning a pretrained backbone at full
        # rate from step zero destroys pretrained features before the randomly
        # initialised head produces a useful gradient.
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_at)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_map = -1.0
    best_epoch = -1
    history: list[dict] = []
    since_improved = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        started = time.time()
        for images, targets in train_loader:
            images, targets = images.to(resolved), targets.to(resolved)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            running += float(loss.item())

        model.eval()
        scores: list[list[float]] = []
        truths: list[list[int]] = []
        with torch.no_grad():
            for images, targets in val_loader:
                probabilities = torch.sigmoid(model(images.to(resolved)))
                scores.extend(probabilities.cpu().tolist())
                truths.extend(targets.int().tolist())

        # mAP, never accuracy. With 14 mostly-absent classes a model that
        # predicts "absent" everywhere exceeds 90% accuracy and is worthless.
        # `mean_average_precision` returns NaN for classes with no positives
        # and excludes them from the mean, rather than scoring them 0 — six of
        # these classes never occur, and averaging in six zeroes would report
        # less than half the real value.
        val_map, per_class_ap = mean_average_precision(scores, truths)
        if math.isnan(val_map):
            val_map = 0.0
        scored_classes = [ap for ap in per_class_ap if not math.isnan(ap)]

        entry = {
            "epoch": epoch,
            "train_loss": round(running / steps_per_epoch, 4),
            "val_mAP": round(val_map, 4),
            "seconds": round(time.time() - started, 1),
            "classes_scored": len(scored_classes),
            "per_class_ap": {
                DETECTION_CLASSES[i]: round(ap, 4)
                for i, ap in enumerate(per_class_ap)
                if not math.isnan(ap)
            },
        }
        history.append(entry)
        logger.info(
            "epoch %2d │ loss %.4f │ val mAP %.4f │ %.0fs",
            epoch, entry["train_loss"], val_map, entry["seconds"],
        )

        if val_map > best_map:
            best_map, best_epoch, since_improved = val_map, epoch, 0
            save_checkpoint(
                out_dir / "tool_best.pt",
                model,
                {
                    "classes": list(DETECTION_CLASSES),
                    "task": "tool_presence",
                    "val_mAP": round(val_map, 4),
                    "epoch": epoch,
                    "trained_on": "cat1 exported frames (boxes → presence)",
                    "provenance_note": (
                        "Trained on five clips of the public SurgVU cat1 set, "
                        "validated on two held-out clips. Only 8 of 14 classes "
                        "occur in this data. Educational and research use only."
                    ),
                },
            )
            logger.info("  ↳ new best (mAP %.4f) → %s", val_map, out_dir / "tool_best.pt")
        else:
            since_improved += 1
            if since_improved >= patience:
                logger.info("early stopping — no improvement in %d epochs", patience)
                break

    summary = {
        "task": "tool_presence",
        "model": model_name,
        "data": str(data_dir),
        "device": resolved,
        "image_size": image_size,
        "epochs_run": len(history),
        "best_val_mAP": round(best_map, 4),
        "best_epoch": best_epoch,
        "checkpoint": str(out_dir / "tool_best.pt"),
        "classes": list(DETECTION_CLASSES),
        "classes_with_positives": present_classes,
        "history": history,
    }
    (out_dir / "presence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def smoke_test() -> int:
    """Full loop on eight synthetic frames — no dataset, ~1 minute."""
    import shutil
    import tempfile

    import numpy as np

    try:
        import cv2
    except ImportError as exc:
        print(f"smoke test needs opencv: {exc}")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="charlie_presence_smoke_"))
    try:
        rng = np.random.default_rng(0)
        for split, count in (("train", 8), ("val", 4)):
            (tmp / "images" / split).mkdir(parents=True)
            (tmp / "labels" / split).mkdir(parents=True)
            for i in range(count):
                cv2.imwrite(
                    str(tmp / "images" / split / f"{i:04d}.jpg"),
                    rng.integers(0, 255, (64, 64, 3), dtype=np.uint8),
                )
                (tmp / "labels" / split / f"{i:04d}.txt").write_text(
                    f"{11 if i % 2 else 2} 0.5 0.5 0.3 0.3\n", encoding="utf-8"
                )
        summary = train(
            tmp, model_name="convnextv2_atto.fcmae_ft_in1k", epochs=1,
            batch_size=2, image_size=64, workers=0, device="cpu",
            out_dir=tmp / "out",
        )
        print("\nSmoke test passed.")
        print(f"  checkpoint : {summary['checkpoint']}")
        print(f"  val mAP    : {summary['best_val_mAP']}")

        # The checkpoint must be loadable by the serving path, and must expose
        # a Grad-CAM layer — that is the entire point of training it.
        from ai.training.models import load_checkpoint

        model, state = load_checkpoint(summary["checkpoint"], task="tool", device="cpu")
        layer = model.gradcam_layer()
        print(f"  reloads    : yes ({len(state.get('classes', []))} classes)")
        print(f"  gradcam    : {'resolved' if layer is not None else 'NO LAYER'}")
        return 0 if layer is not None else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="convnextv2_atto.fcmae_ft_in1k")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s │ %(message)s")

    if args.smoke_test:
        return smoke_test()

    summary = train(
        args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.lr,
        workers=args.workers,
        device=args.device,
        out_dir=args.out,
        threads=args.threads,
        patience=args.patience,
    )
    print("\nPresence training complete")
    print(f"  best val mAP : {summary['best_val_mAP']} (epoch {summary['best_epoch']})")
    print(f"  checkpoint   : {summary['checkpoint']}")
    print("\nGrad-CAM is now available. Press 'Reload models' in /console,")
    print("or POST /api/inference/reload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
