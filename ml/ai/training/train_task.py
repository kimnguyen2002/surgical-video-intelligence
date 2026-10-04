"""
Surgical task recognition — what is being *done*, not just which tool is held.

Run from the repository root::

    python -m ai.training.extract_task_frames --cases 200 --per-interval 15
    python -m ai.training.train_task --epochs 12
    python -m ai.training.train_task --smoke-test

This is the model the platform was missing. Tool presence answers "a needle
driver is mounted"; this answers "the surgeon is suturing" — the question a
trainee actually asks of a recording, and the one `tasks.csv` has labelled all
along across 155 cases.

Single-label, not multi-label
-----------------------------
A frame belongs to one task. So: softmax with cross-entropy, and **balanced
accuracy** selects the checkpoint rather than plain accuracy. Suturing dominates
the label set; a model that predicts it unconditionally scores well on accuracy
and has learned nothing. Balanced accuracy averages per-class recall, so
ignoring a rare class costs exactly as much as ignoring a common one.

Splits are by case
------------------
Frames from one operation share patient, lighting, camera, and instrument set.
A random frame split puts near-duplicates of validation frames into training and
reports a number that collapses on genuinely unseen cases. `split_cases`
partitions case ids with a fixed seed, so the reported accuracy is over
operations the model has never seen.

Label smoothing
---------------
Task boundaries in SurgVU are coarse and transitions are frequently unlabelled.
`extract_task_frames` already avoids sampling near boundaries, but a residual
amount of "this frame is genuinely partly both" remains, and training it to
100% confidence is training on noise.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from ai.training.config import REPO_ROOT, TASK_CLASSES, TASK_DISPLAY

logger = logging.getLogger("surgvu.train_task")

DEFAULT_DATA = REPO_ROOT / "data" / "surgvu_tasks"
DEFAULT_OUT = REPO_ROOT / "checkpoints"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TaskFrameDataset:
    """Single-label surgical task frames, from the extraction manifest."""

    def __init__(self, root: Path, rows: list[dict], train: bool, image_size: int):
        from ai.training.dataset import default_transform

        self.root = Path(root)
        self.rows = rows
        self.transform = default_transform(train=train, image_size=image_size)
        self.classes = list(TASK_CLASSES)
        self.index = {name: i for i, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import torch
        from PIL import Image

        row = self.rows[i]
        with Image.open(self.root / row["file"]) as handle:
            image = handle.convert("RGB")
        return self.transform(image), torch.tensor(
            self.index[row["label"]], dtype=torch.long
        )

    def counts(self) -> Counter:
        return Counter(row["label"] for row in self.rows)


def split_cases(
    rows: list[dict], val_fraction: float = 0.25, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Partition by case id — never by frame."""
    cases = sorted({row["case"] for row in rows})
    if len(cases) < 2:
        return rows, []

    rng = random.Random(seed)
    shuffled = cases[:]
    rng.shuffle(shuffled)
    n_val = max(1, min(len(shuffled) - 1, round(len(shuffled) * val_fraction)))
    val_cases = set(shuffled[:n_val])

    train = [r for r in rows if r["case"] not in val_cases]
    val = [r for r in rows if r["case"] in val_cases]
    return train, val


def balanced_accuracy(confusion: list[list[int]]) -> tuple[float, dict]:
    """Mean per-class recall, ignoring classes absent from the reference set."""
    per_class = {}
    recalls = []
    for i, row in enumerate(confusion):
        total = sum(row)
        if total == 0:
            continue
        recall = row[i] / total
        per_class[TASK_CLASSES[i]] = {
            "recall": round(recall, 4), "support": total, "correct": row[i],
        }
        recalls.append(recall)
    return (sum(recalls) / len(recalls) if recalls else 0.0), per_class


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    data_dir: Path = DEFAULT_DATA,
    *,
    model_name: str = "convnextv2_atto.fcmae_ft_in1k",
    epochs: int = 12,
    batch_size: int = 16,
    image_size: int = 192,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.05,
    workers: int = 0,
    device: str = "auto",
    out_dir: Path = DEFAULT_OUT,
    threads: Optional[int] = None,
    patience: int = 5,
    val_fraction: float = 0.25,
    seed: int = 42,
    label_smoothing: float = 0.1,
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from ai.training.models import SurgVUToolModel, save_checkpoint
    from ai.training.train_detection import configure_cpu_threads

    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"No manifest at {manifest_path}.\n"
            "Build it first:  python -m ai.training.extract_task_frames"
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [r for r in payload["manifest"] if (data_dir / r["file"]).exists()]
    if len(rows) < 40:
        raise SystemExit(f"Only {len(rows)} usable frames — extract more first.")

    resolved = (
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    if resolved == "cpu":
        configure_cpu_threads(threads)

    train_rows, val_rows = split_cases(rows, val_fraction=val_fraction, seed=seed)
    train_set = TaskFrameDataset(data_dir, train_rows, True, image_size)
    val_set = TaskFrameDataset(data_dir, val_rows, False, image_size)

    train_counts = train_set.counts()
    logger.info(
        "train %d frames / %d cases · val %d frames / %d cases",
        len(train_set), len({r["case"] for r in train_rows}),
        len(val_set), len({r["case"] for r in val_rows}),
    )
    for name in TASK_CLASSES:
        logger.info("  %-36s %5d", TASK_DISPLAY.get(name, name), train_counts.get(name, 0))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=workers
    )

    model = SurgVUToolModel(
        model_name=model_name, num_classes=len(TASK_CLASSES), pretrained=True
    ).to(resolved)

    # Inverse-frequency class weights, capped. Suturing outnumbers range-of-
    # motion by an order of magnitude; unweighted cross-entropy is minimised
    # by predicting the majority class and never recovering the rare ones.
    weights = []
    total = max(1, sum(train_counts.values()))
    for name in TASK_CLASSES:
        count = train_counts.get(name, 0)
        weights.append(min(8.0, total / (len(TASK_CLASSES) * count)) if count else 0.0)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=resolved)

    criterion = torch.nn.CrossEntropyLoss(
        weight=weight_tensor, label_smoothing=label_smoothing
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    steps = max(1, len(train_loader))
    total_steps = epochs * steps
    warmup = min(steps, total_steps // 10)

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_at)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_epoch = -1
    history: list[dict] = []
    stale = 0

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
        n = len(TASK_CLASSES)
        confusion = [[0] * n for _ in range(n)]
        correct = seen = 0
        with torch.no_grad():
            for images, targets in val_loader:
                predicted = model(images.to(resolved)).argmax(dim=1).cpu()
                for truth, guess in zip(targets.tolist(), predicted.tolist()):
                    confusion[truth][guess] += 1
                    correct += int(truth == guess)
                    seen += 1

        accuracy = correct / seen if seen else 0.0
        balanced, per_class = balanced_accuracy(confusion)

        entry = {
            "epoch": epoch,
            "train_loss": round(running / steps, 4),
            "val_accuracy": round(accuracy, 4),
            "val_balanced_accuracy": round(balanced, 4),
            "seconds": round(time.time() - started, 1),
            "per_class": per_class,
        }
        history.append(entry)
        logger.info(
            "epoch %2d │ loss %.4f │ acc %.3f │ balanced %.3f │ %.0fs",
            epoch, entry["train_loss"], accuracy, balanced, entry["seconds"],
        )

        if balanced > best_score:
            best_score, best_epoch, stale = balanced, epoch, 0
            save_checkpoint(
                out_dir / "task_best.pt",
                model,
                {
                    "classes": list(TASK_CLASSES),
                    "task": "surgical_task",
                    "val_balanced_accuracy": round(balanced, 4),
                    "val_accuracy": round(accuracy, 4),
                    "epoch": epoch,
                    "image_size": image_size,
                    "train_cases": sorted({r["case"] for r in train_rows}),
                    "val_cases": sorted({r["case"] for r in val_rows}),
                    "provenance_note": (
                        "Surgical task recognition trained on SurgVU tasks.csv "
                        "intervals, split by case. Educational and research use "
                        "only — not a medical device."
                    ),
                },
            )
            logger.info("  ↳ new best (balanced %.4f) → %s", balanced, out_dir / "task_best.pt")
        else:
            stale += 1
            if stale >= patience:
                logger.info("early stopping — no improvement in %d epochs", patience)
                break

    summary = {
        "task": "surgical_task",
        "model": model_name,
        "image_size": image_size,
        "device": resolved,
        "epochs_run": len(history),
        "best_val_balanced_accuracy": round(best_score, 4),
        "best_epoch": best_epoch,
        "checkpoint": str(out_dir / "task_best.pt"),
        "classes": list(TASK_CLASSES),
        "train_frames": len(train_set),
        "val_frames": len(val_set),
        "history": history,
    }
    (out_dir / "task_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def smoke_test() -> int:
    """Whole loop on synthetic frames — no dataset, about a minute."""
    import shutil
    import tempfile

    import numpy as np

    try:
        import cv2
    except ImportError as exc:
        print(f"smoke test needs opencv: {exc}")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="charlie_task_smoke_"))
    try:
        rng = np.random.default_rng(0)
        manifest = []
        # Enough frames to clear the minimum-size guard and to leave both
        # splits non-empty after partitioning by case.
        for case in ("001", "002", "003", "004", "005", "006"):
            for label in TASK_CLASSES[:4]:
                (tmp / label).mkdir(parents=True, exist_ok=True)
                for i in range(4):
                    name = f"{label}/case{case}_{i:04d}.jpg"
                    cv2.imwrite(
                        str(tmp / name),
                        rng.integers(0, 255, (64, 64, 3), dtype=np.uint8),
                    )
                    manifest.append(
                        {
                            "file": name, "case": case, "part": 1,
                            "timestamp": float(i), "label": label,
                        }
                    )
        (tmp / "manifest.json").write_text(
            json.dumps({"manifest": manifest}), encoding="utf-8"
        )

        summary = train(
            tmp, epochs=1, batch_size=2, image_size=64, workers=0,
            device="cpu", out_dir=tmp / "out",
        )
        print("\nSmoke test passed.")
        print(f"  checkpoint : {summary['checkpoint']}")

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
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.25)
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
        val_fraction=args.val_fraction,
    )
    print("\nTask recognition training complete")
    print(f"  balanced accuracy : {summary['best_val_balanced_accuracy']} "
          f"(epoch {summary['best_epoch']})")
    print(f"  checkpoint        : {summary['checkpoint']}")
    best = summary["history"][summary["best_epoch"] - 1] if summary["history"] else {}
    for name, stats in (best.get("per_class") or {}).items():
        print(f"    {TASK_DISPLAY.get(name, name):36s} recall {stats['recall']:.3f} "
              f"(n={stats['support']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
