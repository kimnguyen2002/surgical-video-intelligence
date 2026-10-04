"""
Evaluate a trained SurgVU checkpoint and produce a reproducible report.

    python -m ai.training.evaluate --task tool --checkpoint checkpoints/tool_best.pt
    python -m ai.training.evaluate --task step --checkpoint checkpoints/step_best.pt --split test
    python -m ai.training.evaluate --task tool --checkpoint ... --gradcam-samples 12

Outputs a JSON report, a human-readable table, and (for tool detection)
Grad-CAM overlays for a sample of frames so the numbers can be inspected
visually rather than trusted blindly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ai.training.config import config
from ai.training.metrics import format_report, multiclass_report, multilabel_report
from ai.training.models import load_checkpoint
from ai.training.train_tool_detection import resolve_device

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s │ %(levelname)-5s │ %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("surgvu.evaluate")


@torch.no_grad()
def evaluate_tool(model, loader, classes, device, tune_thresholds: bool) -> dict:
    scores: list[list[float]] = []
    targets_all: list[list[int]] = []
    for images, targets in loader:
        logits = model(images.to(device))
        scores.extend(torch.sigmoid(logits).float().cpu().tolist())
        targets_all.extend(targets.int().cpu().tolist())
    return multilabel_report(scores, targets_all, classes, tune_thresholds=tune_thresholds)


@torch.no_grad()
def evaluate_step(model, loader, classes, device) -> dict:
    predictions: list[int] = []
    targets_all: list[int] = []
    probabilities: list[list[float]] = []
    for clips, targets in loader:
        logits = model(clips.to(device))
        probabilities.extend(torch.softmax(logits.float(), dim=1).cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets_all.extend(targets.cpu().tolist())
    return multiclass_report(predictions, targets_all, classes, probabilities)


def render_gradcams(model, dataset, classes, device, count: int, out_dir: Path) -> list[str]:
    """Save Grad-CAM overlays for the highest-confidence prediction per frame."""
    from PIL import Image

    from ai.explainability import GradCAM, heatmap_to_png

    layer = model.gradcam_layer()
    if layer is None:
        logger.warning("No Grad-CAM target layer for this architecture — skipping.")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    step = max(len(dataset) // max(count, 1), 1)

    with GradCAM(model, layer) as cam:
        for n, index in enumerate(range(0, len(dataset), step)):
            if n >= count:
                break
            tensor, target = dataset[index]
            batch = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                probabilities = torch.sigmoid(model(batch))[0]
            class_index = int(probabilities.argmax().item())

            heatmap = cam.generate(batch, class_idx=class_index)
            if heatmap is None:
                continue

            record, _ = dataset.samples[index]
            try:
                source = Image.open(record.path).convert("RGB")
            except OSError:
                continue

            data_url = heatmap_to_png(source, heatmap, alpha=0.55)
            if not data_url:
                continue

            import base64

            name = (
                f"{n:02d}_{classes[class_index]}"
                f"_p{probabilities[class_index]:.2f}"
                f"_{'TP' if target[class_index] else 'FP'}.png"
            )
            path = out_dir / name
            path.write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
            written.append(str(path))

    logger.info("Wrote %d Grad-CAM overlays → %s", len(written), out_dir)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a SurgVU checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=["tool", "step"], default="tool")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--workers", type=int, default=config.num_workers)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tune-thresholds", action="store_true",
                        help="Tune per-class thresholds (use on val, not test).")
    parser.add_argument("--gradcam-samples", type=int, default=8)
    parser.add_argument("--output", default="", help="Report path (default: next to the checkpoint).")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        logger.error(
            "Checkpoint not found: %s\nTrain one first:\n"
            "  python -m ai.training.train_%s",
            checkpoint,
            "tool_detection" if args.task == "tool" else "step_recognition",
        )
        return 1

    device = resolve_device(args.device)
    model, state = load_checkpoint(checkpoint, task=args.task, device=str(device))
    classes = state.get(
        "classes", config.tool_classes if args.task == "tool" else config.task_classes
    )
    logger.info(
        "Loaded %s (epoch %s) — evaluating on '%s'",
        checkpoint.name, state.get("epoch", "?"), args.split,
    )

    if args.task == "tool":
        from ai.training.dataset import SurgVUToolDataset

        dataset = SurgVUToolDataset(split=args.split)
        if not len(dataset):
            logger.error("Split '%s' is empty.", args.split)
            return 1
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
        report = evaluate_tool(model, loader, classes, device, args.tune_thresholds)
        title = f"SurgVU Tool Detection — {args.split}"
    else:
        from ai.training.dataset import SurgVUTaskDataset

        dataset = SurgVUTaskDataset(split=args.split, clip_length=state.get("clip_length"))
        if not len(dataset):
            logger.error("Split '%s' is empty.", args.split)
            return 1
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
        report = evaluate_step(model, loader, classes, device)
        title = f"SurgVU Step Recognition — {args.split}"

    print(format_report(report, title))

    if args.task == "step":
        print("\nConfusion matrix (rows = truth, cols = prediction):")
        header = "".join(f"{n[:9]:>11}" for n in classes)
        print(f"{'':<34}{header}")
        for name, row in zip(classes, report["confusion_matrix"]):
            print(f"{name[:33]:<34}" + "".join(f"{v:>11}" for v in row))

    report["checkpoint"] = str(checkpoint)
    report["split"] = args.split
    report["task"] = args.task
    report["source_epoch"] = state.get("epoch")

    output = Path(args.output) if args.output else checkpoint.parent / f"{args.task}_{args.split}_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Report written → %s", output)

    if args.task == "tool" and args.gradcam_samples > 0:
        render_gradcams(
            model, dataset, classes, device, args.gradcam_samples,
            checkpoint.parent / "gradcam" / args.split,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
