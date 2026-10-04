"""
Supervised instrument detection on the cat1 boxes.

Run from the repository root::

    python -m ai.training.export_detection          # once, to build the dataset
    python -m ai.training.train_detection --epochs 40
    python -m ai.training.train_detection --smoke-test

This is the **spatial** half of the platform's perception. The other half —
tool presence over all 155 surgvu24 cases — is
:mod:`ai.training.train_tool_detection`, and the weakly-supervised bridge
between them is :mod:`ai.training.localize`.

What this model is, stated plainly
----------------------------------
It is trained on five clips of the *public cat1 test set* and validated on two
held-out clips. Six of the fourteen classes never appear in the data at all, so
they can be neither learned nor scored. It is a small-data demonstrator that
puts real, labelled boxes on screen — not a validated detector, and every
surface that renders its output says so.

Why Ultralytics rather than a hand-rolled head
----------------------------------------------
Detection needs anchor assignment, NMS, mosaic augmentation, and a matched
loss; reimplementing those is a large surface area for silent bugs that show up
only as a quietly worse mAP. The dependency is optional and confined to this
one module — nothing in the serving path imports it, so an install without
Ultralytics still runs the platform and simply has no detector.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from ai.datasets.coco import DETECTION_CLASSES
from ai.training.config import REPO_ROOT, config

logger = logging.getLogger("surgvu.train_detection")

DEFAULT_DATA = REPO_ROOT / "data" / "cat1_yolo" / "dataset.yaml"
DEFAULT_OUT = REPO_ROOT / "checkpoints"


def configure_cpu_threads(threads: int | None = None) -> int:
    """
    Give PyTorch every core it is allowed to use.

    On this class of machine ``torch.get_num_threads()`` comes back as 1 when
    ``OMP_NUM_THREADS`` is unset or inherited as 1, which makes CPU training
    roughly fourfold slower than the hardware permits — and it fails silently,
    looking only like "training is slow". Both the environment variable and the
    runtime setting are needed: OpenMP reads the variable at first use, which
    may already have happened by the time this runs.
    """
    import torch  # noqa: PLC0415

    if threads is None:
        threads = max(1, (os.cpu_count() or 2))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    try:
        torch.set_num_threads(threads)
    except (RuntimeError, ValueError) as exc:  # pragma: no cover
        logger.debug("could not set torch threads: %s", exc)
    effective = torch.get_num_threads()
    logger.info("CPU threads: %d (requested %d)", effective, threads)
    return effective


def _repoint_dataset_yaml(yaml_path: Path) -> None:
    """
    Rewrite the dataset's ``path:`` to wherever it actually lives now.

    Ultralytics needs an absolute root, so ``export_detection`` writes one. That
    makes the file non-portable: copy the repository to another machine — a
    different drive letter, or macOS — and training fails with "no labels
    found" while pointing at a path that does not exist. Since the YAML always
    sits at the root of its own dataset, the correct value is derivable, so it
    is corrected in place on every run instead of trusted.
    """
    directory = yaml_path.parent.resolve()
    try:
        lines = yaml_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    wanted = f"path: {directory.as_posix()}"
    changed = False
    for index, line in enumerate(lines):
        if line.startswith("path:"):
            if line.strip() != wanted:
                lines[index] = wanted
                changed = True
            break
    else:
        lines.insert(0, wanted)
        changed = True

    if changed:
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Repointed %s at %s", yaml_path.name, directory)


def _resolve_device(requested: str) -> str:
    import torch  # noqa: PLC0415

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def train(
    data: Path = DEFAULT_DATA,
    *,
    model: str = "yolo11n.pt",
    epochs: int = 40,
    imgsz: int = 512,
    batch: int = 8,
    device: str = "auto",
    workers: int = 2,
    patience: int = 10,
    out_dir: Path = DEFAULT_OUT,
    run_name: str = "cat1_detect",
    resume: bool = False,
    threads: int | None = None,
) -> dict:
    """Fine-tune a YOLO detector on the exported cat1 dataset."""
    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Ultralytics is required for detection training.\n"
            "    pip install ultralytics"
        ) from exc

    data = Path(data)
    if not data.exists():
        raise SystemExit(
            f"No dataset at {data}.\n"
            "Build it first:  python -m ai.training.export_detection"
        )
    _repoint_dataset_yaml(data)

    device = _resolve_device(device)
    if device == "cpu":
        configure_cpu_threads(threads)
        logger.warning(
            "Training on CPU. This is viable for this dataset (~5k frames) but "
            "slow; expect hours rather than minutes."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLO(model)
    results = detector.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        project=str(out_dir / "detection"),
        name=run_name,
        exist_ok=True,
        resume=resume,
        # Surgical-video augmentation. Horizontal flips are safe: the endoscope
        # has no preferred handedness. Vertical flips and large rotations are
        # not — endoscopic video has a consistent horizon, and training on
        # upside-down frames spends capacity on an orientation that never
        # occurs in the data or at inference.
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        # Mosaic helps a small dataset see more context per step, but it
        # fabricates instrument co-occurrences that never happen in one
        # operation, so it is turned off for the final epochs.
        mosaic=1.0,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        translate=0.1,
        scale=0.4,
        erasing=0.2,
        plots=True,
        verbose=True,
    )

    run_dir = Path(results.save_dir) if getattr(results, "save_dir", None) else out_dir
    best = run_dir / "weights" / "best.pt"

    summary = {
        "task": "detection",
        "model": model,
        "data": str(data),
        "device": device,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "run_dir": str(run_dir),
        "best_weights": str(best) if best.exists() else None,
        "classes": DETECTION_CLASSES,
    }

    metrics = getattr(results, "results_dict", None) or {}
    summary["metrics"] = {k: _to_float(v) for k, v in metrics.items()}

    # A stable path the serving layer can look for without knowing run names.
    if best.exists():
        stable = out_dir / "detection_best.pt"
        stable.write_bytes(best.read_bytes())
        summary["stable_weights"] = str(stable)

    (run_dir / "charlie_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _register_experiment(summary)
    return summary


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _register_experiment(summary: dict) -> None:
    """Record the run alongside the classification runs, if tracking is present."""
    try:
        from ai.evaluation import experiments  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - tracking is optional
        logger.debug("experiment tracking unavailable: %s", exc)
        return
    for fn_name in ("record_run", "register_run", "log_run"):
        fn = getattr(experiments, fn_name, None)
        if callable(fn):
            try:
                fn(summary)
            except Exception as exc:  # pragma: no cover
                logger.debug("experiment tracking failed: %s", exc)
            return


def smoke_test() -> int:
    """
    Run the whole loop on a handful of synthetic frames.

    Confirms the dependency, the YAML, the label format, and the training and
    validation passes without needing the dataset or hours of compute.
    """
    import shutil
    import tempfile

    import numpy as np

    try:
        import cv2
        from ultralytics import YOLO  # noqa: F401
    except ImportError as exc:
        print(f"smoke test needs opencv + ultralytics: {exc}")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="charlie_det_smoke_"))
    try:
        rng = np.random.default_rng(0)
        for split, count in (("train", 8), ("val", 4)):
            (tmp / "images" / split).mkdir(parents=True)
            (tmp / "labels" / split).mkdir(parents=True)
            for i in range(count):
                image = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
                cv2.imwrite(str(tmp / "images" / split / f"{i:04d}.jpg"), image)
                (tmp / "labels" / split / f"{i:04d}.txt").write_text(
                    "11 0.5 0.5 0.3 0.3\n", encoding="utf-8"
                )
        yaml_path = tmp / "dataset.yaml"
        yaml_path.write_text(
            f"path: {tmp.as_posix()}\ntrain: images/train\nval: images/val\n"
            f"nc: {len(DETECTION_CLASSES)}\nnames:\n"
            + "".join(f"  {i}: {n}\n" for i, n in enumerate(DETECTION_CLASSES)),
            encoding="utf-8",
        )
        summary = train(
            yaml_path,
            epochs=1,
            imgsz=128,
            batch=2,
            device="cpu",
            workers=0,
            out_dir=tmp / "out",
            run_name="smoke",
        )
        print("\nSmoke test passed.")
        print(f"  weights: {summary.get('best_weights')}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11n.pt", help="yolo11n/s/m .pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto | cpu | 0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--name", default="cat1_detect")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s │ %(message)s")

    if args.smoke_test:
        return smoke_test()

    summary = train(
        args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        out_dir=args.out,
        run_name=args.name,
        resume=args.resume,
        threads=args.threads,
    )
    print("\nDetection training complete")
    print(f"  run     : {summary['run_dir']}")
    print(f"  weights : {summary.get('stable_weights') or summary.get('best_weights')}")
    for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)",
                "metrics/recall(B)"):
        if key in summary.get("metrics", {}):
            print(f"  {key:22s}: {summary['metrics'][key]:.4f}")
    print(
        "\n⚕ Trained on the public cat1 clips — a small-data demonstrator for "
        "education and research, not a validated detector."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
