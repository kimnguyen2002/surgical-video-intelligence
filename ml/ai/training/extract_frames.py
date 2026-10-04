"""
Decode SurgVU videos into sampled frames plus a manifest.

The dataset is 720p at 60 fps across ~840 hours. Training directly from video
would spend nearly all its time in the decoder, and consecutive 60 fps frames
are near-duplicates that add compute without adding information. So videos are
decoded once at a low sampling rate (1 fps by default) into JPEGs, and every
later stage reads the manifest.

The manifest records each frame's ``(case, part, timestamp)``, which is what
makes interval labels resolvable, and each part's duration, which is what makes
multi-part label intervals resolvable.

Usage::

    python -m ai.training.extract_frames --dataset data/surgvu --fps 1
    python -m ai.training.extract_frames --dataset data/surgvu --limit 3   # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ai.training.config import config
from ai.training.labels import parse_case_part

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s │ %(levelname)-5s │ %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("surgvu.extract")

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
MANIFEST_NAME = "manifest.json"


def find_videos(dataset_dir: Path) -> list[Path]:
    return sorted(
        p for p in Path(dataset_dir).rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES
    )


def _extract_with_opencv(video: Path, out_dir: Path, fps: float, quality: int) -> list[dict]:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(source_fps / fps)))

    frames: list[dict] = []
    index = 0
    saved = 0
    while True:
        ok = capture.grab()  # grab() skips decode; far cheaper than read()
        if not ok:
            break
        if index % step == 0:
            ok, image = capture.retrieve()
            if ok:
                timestamp = index / source_fps
                name = f"{saved:07d}.jpg"
                cv2.imwrite(
                    str(out_dir / name), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                )
                frames.append({"file": name, "timestamp": round(timestamp, 3)})
                saved += 1
        index += 1

    capture.release()
    duration = (total / source_fps) if total else (frames[-1]["timestamp"] if frames else 0.0)
    for frame in frames:
        frame["duration"] = round(duration, 3)
    return frames


def _extract_with_ffmpeg(video: Path, out_dir: Path, fps: float, quality: int) -> list[dict]:
    # ffmpeg's -q:v runs 2 (best) to 31 (worst); map from JPEG quality.
    qscale = max(2, min(31, round((100 - quality) / 3)))
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video),
            "-vf", f"fps={fps}", "-q:v", str(qscale),
            str(out_dir / "%07d.jpg"),
        ],
        check=True,
    )
    files = sorted(out_dir.glob("*.jpg"))
    duration = _probe_duration(video) or (len(files) / fps)
    return [
        {
            "file": path.name,
            "timestamp": round(i / fps, 3),
            "duration": round(duration, 3),
        }
        for i, path in enumerate(files)
    ]


def _probe_duration(video: Path) -> Optional[float]:
    if not shutil.which("ffprobe"):
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video),
            ],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return None


def extract_video(
    video: Path, frames_root: Path, fps: float, quality: int = 88, force: bool = False
) -> Optional[dict]:
    """Extract one video. Returns its manifest entry, or None on failure."""
    case, part = parse_case_part(video.stem)
    key = f"case_{case}_part_{part:03d}"
    out_dir = frames_root / key

    done_marker = out_dir / ".complete"
    if done_marker.exists() and not force:
        try:
            return json.loads(done_marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass  # Re-extract if the marker is unreadable.

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jpg"):
        stale.unlink()

    try:
        try:
            import cv2  # noqa: F401

            frames = _extract_with_opencv(video, out_dir, fps, quality)
        except ImportError:
            if not shutil.which("ffmpeg"):
                raise RuntimeError(
                    "Neither OpenCV nor ffmpeg is available. Install one:\n"
                    "  pip install opencv-python-headless\n"
                    "  brew install ffmpeg   |   apt-get install ffmpeg"
                )
            frames = _extract_with_ffmpeg(video, out_dir, fps, quality)
    except Exception as exc:
        logger.error("Failed to extract %s: %s", video.name, exc)
        return None

    if not frames:
        logger.warning("No frames extracted from %s", video.name)
        return None

    entry = {
        "case": case,
        "part": part,
        "key": key,
        "source": str(video),
        "dir": str(out_dir),
        "fps": fps,
        "count": len(frames),
        "duration": frames[0].get("duration", frames[-1]["timestamp"]),
        "frames": frames,
    }
    done_marker.write_text(json.dumps(entry), encoding="utf-8")
    logger.info("%s → %d frames (%.1fs)", video.name, len(frames), entry["duration"])
    return entry


def build_manifest(
    dataset_dir: Path,
    frames_root: Path,
    fps: float,
    limit: Optional[int] = None,
    force: bool = False,
) -> dict:
    videos = find_videos(dataset_dir)
    if not videos:
        raise SystemExit(
            f"No videos found under {dataset_dir}.\n"
            "Download the dataset first:\n"
            "  python -m ai.training.download_dataset --dest data/surgvu --extract"
        )
    if limit:
        videos = videos[:limit]

    logger.info("Extracting %d videos at %.2f fps → %s", len(videos), fps, frames_root)
    frames_root.mkdir(parents=True, exist_ok=True)

    entries = []
    for i, video in enumerate(videos, start=1):
        logger.info("[%d/%d] %s", i, len(videos), video.name)
        entry = extract_video(video, frames_root, fps, force=force)
        if entry:
            entries.append(entry)

    manifest = {
        "fps": fps,
        "dataset_dir": str(dataset_dir),
        "parts": entries,
        "total_frames": sum(e["count"] for e in entries),
        "total_seconds": round(sum(e["duration"] for e in entries), 1),
        "cases": sorted({e["case"] for e in entries}),
    }
    path = frames_root / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "Manifest written: %s (%d parts, %d frames, %.1f hours)",
        path,
        len(entries),
        manifest["total_frames"],
        manifest["total_seconds"] / 3600,
    )
    return manifest


def load_manifest(frames_root: Path) -> dict:
    path = Path(frames_root) / MANIFEST_NAME
    if not path.exists():
        raise SystemExit(
            f"No manifest at {path}.\n"
            "Extract frames first:\n"
            f"  python -m ai.training.extract_frames --dataset {config.dataset_dir} --fps {config.extract_fps}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def part_durations(manifest: dict) -> dict[tuple[str, int], float]:
    """``(case, part) → duration`` — what label resolution needs."""
    return {(e["case"], e["part"]): e["duration"] for e in manifest["parts"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frames from SurgVU videos.")
    parser.add_argument("--dataset", default=config.dataset_dir)
    parser.add_argument("--frames", default=config.frames_dir)
    parser.add_argument("--fps", type=float, default=config.extract_fps)
    parser.add_argument("--limit", type=int, help="Only process the first N videos.")
    parser.add_argument("--force", action="store_true", help="Re-extract completed videos.")
    args = parser.parse_args()

    build_manifest(
        Path(args.dataset), Path(args.frames), args.fps, limit=args.limit, force=args.force
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
