"""
Train the SurgVU surgical step/task recogniser (temporal classification).

    python -m ai.training.train_step_recognition --model cnn_gru
    python -m ai.training.train_step_recognition --model cnn_gru \
        --init-from checkpoints/tool_best.pt          # warm start (recommended)
    python -m ai.training.train_step_recognition --model timesformer --clip-len 8
    python -m ai.training.train_step_recognition --smoke-test

Design notes:

* **Balanced accuracy and macro-F1 select the checkpoint, not accuracy.** The
  ``other`` class dominates SurgVU tasks, so plain accuracy rewards a model
  that predicts ``other`` and gives up on everything else.
* **Class-weighted cross-entropy with label smoothing.** Weights counter the
  imbalance; smoothing counters the coarse task boundaries — a clip straddling
  a transition genuinely is partly both, so a model trained to be 100%
  confident there is being trained on noise.
* **Warm-starting from the tool detector** usually beats ImageNet
  initialisation: tool detection has far more supervision and learns exactly
  the surgical features this task needs.
* **Frozen-encoder warmup.** For the first epochs only the temporal head
  trains, so the randomly-initialised GRU cannot wash out pretrained features
  with large early gradients.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai.evaluation import experiment_tracker
from ai.training.config import config
from ai.training.metrics import format_report, multiclass_report
from ai.training.models import SurgVUStepModel, save_checkpoint
from ai.training.train_tool_detection import build_scheduler, resolve_device

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s │ %(levelname)-5s │ %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("surgvu.train.step")


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    predictions: list[int] = []
    targets_all: list[int] = []
    probabilities: list[list[float]] = []

    for clips, targets in loader:
        clips = clips.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(clips)
        total_loss += criterion(logits, targets).item()
        probs = torch.softmax(logits.float(), dim=1)
        probabilities.extend(probs.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets_all.extend(targets.cpu().tolist())

    report = multiclass_report(predictions, targets_all, class_names, probabilities)
    return total_loss / max(len(loader), 1), report


def set_encoder_trainable(model: SurgVUStepModel, trainable: bool) -> None:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return
    for parameter in encoder.parameters():
        parameter.requires_grad = trainable


def smoke_test(args) -> int:
    """Verify the temporal loop on synthetic clips — no dataset required."""
    from torch.utils.data import TensorDataset

    device = resolve_device(args.device)
    classes = config.task_classes
    logger.info("SMOKE TEST — synthetic clips, model=%s, device=%s", args.model, device)

    n, size = 8, config.image_size
    clips = torch.randn(n, args.clip_len, 3, size, size)
    targets = torch.randint(0, len(classes), (n,))
    loader = DataLoader(TensorDataset(clips, targets), batch_size=2)

    model = SurgVUStepModel(
        model_name=args.model,
        num_classes=len(classes),
        frame_encoder=args.encoder,
        pretrained=False,
        clip_length=args.clip_len,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for clip_batch, target_batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(clip_batch.to(device)), target_batch.to(device))
        loss.backward()
        optimizer.step()
    logger.info("Training step OK (final batch loss %.4f)", loss.item())

    _, report = evaluate(model, loader, criterion, device, classes)
    logger.info(
        "Metrics OK — accuracy=%s balanced=%s macro_f1=%s",
        report["accuracy"], report["balanced_accuracy"], report["macro_f1"],
    )

    if args.model == "cnn_gru":
        with torch.no_grad():
            _, attention = model(clips[:1].to(device), return_attention=True)
        logger.info("Temporal attention OK — weights shape %s", tuple(attention.shape))

    output = Path(args.output) / "smoke_step.pt"
    save_checkpoint(output, model, {"smoke_test": True, "classes": classes})
    logger.info("Checkpoint OK → %s", output)
    output.unlink(missing_ok=True)

    logger.info("✅ Smoke test passed — the temporal pipeline is wired correctly.")
    return 0


def train(args) -> int:
    from ai.training.dataset import SurgVUTaskDataset

    device = resolve_device(args.device)
    classes = config.task_classes
    logger.info("Device: %s | Model: %s | clip=%d", device, args.model, args.clip_len)

    train_set = SurgVUTaskDataset(split="train", clip_length=args.clip_len)
    val_set = SurgVUTaskDataset(split="val", clip_length=args.clip_len)
    if len(train_set) == 0:
        logger.error("Training split is empty — check the dataset and manifest.")
        return 1

    logger.info("Clip distribution: %s", json.dumps(train_set.describe()["classes"]))

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
    )

    model = SurgVUStepModel(
        model_name=args.model,
        num_classes=len(classes),
        frame_encoder=args.encoder,
        pretrained=not args.no_pretrained,
        clip_length=args.clip_len,
    ).to(device)

    if args.init_from:
        model.load_frame_encoder_from_tool_model(args.init_from)

    weights = train_set.class_weights().to(device)
    logger.info(
        "class weights: %s",
        {name: round(w, 2) for name, w in zip(classes, weights.cpu().tolist())},
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=config.weight_decay
    )
    scheduler = build_scheduler(optimizer, args.epochs, len(train_loader), config.warmup_epochs)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if args.freeze_encoder_epochs > 0:
        set_encoder_trainable(model, False)
        logger.info("Frame encoder frozen for the first %d epochs", args.freeze_encoder_epochs)

    experiment = experiment_tracker.start(
        name=args.name or f"step-{args.model}",
        task="step_recognition",
        model=args.model,
        config={
            **config.as_dict(), "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "clip_length": args.clip_len, "encoder": args.encoder,
            "init_from": args.init_from or None,
        },
        dataset=f"SurgVU@{config.extract_fps}fps",
        notes=args.notes,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, patience = -1.0, -1, 0

    try:
        for epoch in range(args.epochs):
            if epoch == args.freeze_encoder_epochs and args.freeze_encoder_epochs > 0:
                set_encoder_trainable(model, True)
                logger.info("Frame encoder unfrozen")

            model.train()
            running = 0.0
            started = time.time()

            for step, (clips, targets) in enumerate(train_loader):
                clips = clips.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = criterion(model(clips), targets)

                scaler.scale(loss).backward()
                if config.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running += loss.item()
                if step % max(len(train_loader) // 20, 1) == 0:
                    logger.info(
                        "epoch %d/%d  step %d/%d  loss %.4f  lr %.2e",
                        epoch + 1, args.epochs, step, len(train_loader),
                        loss.item(), scheduler.get_last_lr()[0],
                    )
                if args.debug and step >= 2:
                    break

            train_loss = running / max(len(train_loader), 1)
            val_loss, report = evaluate(model, val_loader, criterion, device, classes)

            logger.info(
                "epoch %d — train %.4f | val %.4f | acc %.4f | balanced %.4f | macro-F1 %.4f | %.0fs",
                epoch + 1, train_loss, val_loss, report["accuracy"],
                report["balanced_accuracy"], report["macro_f1"], time.time() - started,
            )
            experiment_tracker.log(
                experiment.id, epoch + 1,
                train_loss=round(train_loss, 5), val_loss=round(val_loss, 5),
                accuracy=report["accuracy"], balanced_accuracy=report["balanced_accuracy"],
                macro_f1=report["macro_f1"],
            )

            # Mean of balanced accuracy and macro-F1: rewards getting the rare
            # tasks right rather than riding the dominant class.
            score = (report["balanced_accuracy"] + report["macro_f1"]) / 2
            if score > best_score:
                best_score, best_epoch, patience = score, epoch + 1, 0
                save_checkpoint(
                    output_dir / "step_best.pt",
                    model,
                    {
                        "task": "step_recognition", "classes": classes, "epoch": epoch + 1,
                        "score": score, "report": report, "frame_encoder": args.encoder,
                        "clip_length": args.clip_len, "config": config.as_dict(),
                    },
                )
                logger.info("  ↳ new best score %.4f — checkpoint saved", score)
            else:
                patience += 1
                if patience >= config.early_stopping_patience:
                    logger.info("Early stopping — no improvement for %d epochs", patience)
                    break

            if args.debug:
                break

        logger.info("Evaluating best checkpoint on the test split…")
        test_set = SurgVUTaskDataset(split="test", clip_length=args.clip_len)
        final = {}
        if len(test_set):
            state = torch.load(output_dir / "step_best.pt", map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers)
            _, final = evaluate(model, test_loader, criterion, device, classes)
            print(format_report(final, "SurgVU Step Recognition — test split"))
            (output_dir / "step_test_report.json").write_text(
                json.dumps(final, indent=2), encoding="utf-8"
            )

        experiment_tracker.finish(
            experiment.id,
            metrics={
                "best_val_score": round(best_score, 4),
                "best_epoch": best_epoch,
                "test_accuracy": final.get("accuracy"),
                "test_balanced_accuracy": final.get("balanced_accuracy"),
                "test_macro_f1": final.get("macro_f1"),
            },
            artifacts={"checkpoint": str(output_dir / "step_best.pt")},
        )
        logger.info("Done — best score %.4f at epoch %d", best_score, best_epoch)
        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted — marking experiment as failed.")
        experiment_tracker.finish(experiment.id, status="failed")
        return 130


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SurgVU step recognition.")
    parser.add_argument("--model", default="cnn_gru",
                        choices=["cnn_gru", "mean_pool", "timesformer", "videomae"])
    parser.add_argument("--encoder", default="convnextv2_tiny.fcmae_ft_in22k_in1k",
                        help="Frame encoder for cnn_gru / mean_pool.")
    parser.add_argument("--init-from", default="",
                        help="Tool-detection checkpoint to warm-start the frame encoder from.")
    parser.add_argument("--clip-len", type=int, default=config.clip_length)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--lr", type=float, default=config.learning_rate)
    parser.add_argument("--workers", type=int, default=config.num_workers)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=2)
    parser.add_argument("--output", default=config.output_dir)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true", default=config.amp)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--name", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        return smoke_test(args)
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
