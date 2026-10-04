"""
Decode the cat1 clips and export them as a YOLO detection dataset.

Run from the repository root::

    python -m ai.training.export_detection
    python -m ai.training.export_detection --val-fraction 0.3 --seed 42

The clips ship pre-decoded at 1 fps, so a COCO ``image_id`` is exactly the
frame index — frames are read sequentially and matched by position, with no
timestamp seeking. Seeking per-frame with ``CAP_PROP_POS_FRAMES`` would be both
slower and, on some builds, inexact on B-frames; a linear read is neither.

The split is by **clip**, never by frame. Consecutive 1 fps frames of the same
instrument in the same operation are near-duplicates, so a frame-level split
reports a validation number that is really a training number.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from ai.datasets.coco import (
    DETECTION_CLASSES,
    ClipAnnotations,
    dataset_report,
    load_all,
    split_clips,
)
from ai.training.config import REPO_ROOT

logger = logging.getLogger("surgvu.export_detection")

DEFAULT_OUT = REPO_ROOT / "data" / "cat1_yolo"


def _require_cv2():
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "OpenCV is required to decode the clips.\n"
            "    pip install opencv-python-headless"
        ) from exc
    return cv2


def export_clip(
    clip: ClipAnnotations,
    images_dir: Path,
    labels_dir: Path,
    *,
    overwrite: bool = False,
    stride: int = 1,
) -> int:
    """
    Write one clip's frames and YOLO label files.

    Returns the number of frames written. Frames are named
    ``clip<name>_<index:06d>.jpg`` so that a file is traceable back to its
    source clip and second offset from the filename alone — which matters when
    reviewing a false positive months later.

    ``stride`` keeps every *n*-th annotated frame. At 1 fps, adjacent frames of
    the same operation are near-duplicates: they cost a full training step each
    but carry very little additional information, and they inflate epoch time
    without improving the model. Striding the **training** split is a cheap,
    honest speedup. The validation split must always use ``stride=1`` — a
    metric computed on a thinned set is not comparable to one that is not.
    """
    cv2 = _require_cv2()

    wanted = clip.frames_by_index()
    if stride > 1:
        keep = sorted(wanted)[::stride]
        wanted = {i: wanted[i] for i in keep}
    if not wanted:
        return 0

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(clip.video))
    if not capture.isOpened():
        logger.error("Cannot open %s", clip.video)
        return 0

    written = 0
    position = 0
    last_wanted = max(wanted)
    try:
        while position <= last_wanted:
            ok, frame = capture.read()
            if not ok:
                break
            annotation = wanted.get(position)
            if annotation is not None:
                stem = f"clip{clip.clip}_{position:06d}"
                image_path = images_dir / f"{stem}.jpg"
                label_path = labels_dir / f"{stem}.txt"

                if overwrite or not image_path.exists():
                    cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                height, width = frame.shape[:2]
                # Trust the decoded frame's dimensions over the JSON's: if they
                # ever disagree, the pixels are what the model actually sees.
                lines = []
                for box in annotation.boxes:
                    cx, cy, bw, bh = box.to_yolo(width, height)
                    if bw <= 0 or bh <= 0:
                        continue
                    lines.append(f"{box.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                written += 1
            position += 1
    finally:
        capture.release()

    missing = len(wanted) - written
    if missing > 0:
        logger.warning(
            "clip %s: %d annotated frames were past the end of the video",
            clip.clip, missing,
        )
    return written


def export(
    out_dir: Path = DEFAULT_OUT,
    *,
    val_fraction: float = 0.3,
    seed: int = 42,
    overwrite: bool = False,
    clean: bool = False,
    train_stride: int = 1,
) -> dict:
    """Export every discovered cat1 clip as a YOLO dataset. Returns a manifest."""
    clips = load_all()
    if not clips:
        raise SystemExit(
            "No cat1 clips found. Expected a directory like "
            "'cat1_test_set_public' beside the repository, or set "
            "SURGVU_CAT1_ROOT."
        )

    out_dir = Path(out_dir)
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)

    train_clips, val_clips = split_clips(clips, val_fraction=val_fraction, seed=seed)
    logger.info("train clips: %s", train_clips)
    logger.info("val   clips: %s", val_clips)

    counts: dict[str, int] = {}
    for split, names in (("train", train_clips), ("val", val_clips)):
        # Validation is never thinned — see export_clip's docstring.
        stride = train_stride if split == "train" else 1
        for name in names:
            n = export_clip(
                clips[name],
                out_dir / "images" / split,
                out_dir / "labels" / split,
                overwrite=overwrite,
                stride=stride,
            )
            counts[name] = n
            logger.info("  %s → %s (%d frames, stride %d)", name, split, n, stride)

    # Ultralytics resolves `path` relative to its own settings unless absolute,
    # so write an absolute one and avoid a class of confusing "no labels found".
    yaml_text = (
        "# cat1 surgical instrument detection — generated by "
        "ai/training/export_detection.py\n"
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(DETECTION_CLASSES)}\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(DETECTION_CLASSES))
    )
    (out_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    manifest = {
        "out_dir": str(out_dir.resolve()),
        "train_clips": train_clips,
        "val_clips": val_clips,
        "frames_per_clip": counts,
        "frames_train": sum(counts[c] for c in train_clips),
        "frames_val": sum(counts[c] for c in val_clips),
        "split": {
            "val_fraction": val_fraction,
            "seed": seed,
            "by": "clip",
            "train_stride": train_stride,
        },
        "classes": DETECTION_CLASSES,
        "source_report": dataset_report(clips),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="re-encode existing JPEGs")
    parser.add_argument("--clean", action="store_true", help="delete the output dir first")
    parser.add_argument(
        "--train-stride", type=int, default=1,
        help="keep every n-th training frame (validation is never thinned)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s │ %(message)s")

    manifest = export(
        args.out,
        val_fraction=args.val_fraction,
        seed=args.seed,
        overwrite=args.overwrite,
        clean=args.clean,
        train_stride=args.train_stride,
    )

    report = manifest["source_report"]
    print("\nExported cat1 detection dataset")
    print(f"  output      : {manifest['out_dir']}")
    print(f"  train       : {manifest['frames_train']} frames from clips {manifest['train_clips']}")
    print(f"  val         : {manifest['frames_val']} frames from clips {manifest['val_clips']}")
    print(f"  boxes       : {report['boxes']}")
    print(f"  classes seen: {report['classes_present']}/{len(DETECTION_CLASSES)}")
    if report["classes_absent"]:
        print(f"  never seen  : {', '.join(report['classes_absent'])}")
        print("                (these cannot be learned or scored from this data)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
