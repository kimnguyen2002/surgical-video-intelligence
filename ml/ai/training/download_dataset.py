"""
Download the SurgVU dataset (arXiv:2501.09209v1).

    python -m ai.training.download_dataset --dest data/surgvu --extract
    python -m ai.training.download_dataset --dest data/surgvu --only labels --extract
    python -m ai.training.download_dataset --check          # sizes only, no download

Sizes: the video archive is roughly 100+ GB (840 hours of 720p). Start with
``--only labels`` (a few MB) to inspect the label schema before committing the
disk and bandwidth.

Downloads resume: an interrupted transfer continues from where it stopped
rather than starting over.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)-5s │ %(message)s")
logger = logging.getLogger("surgvu.download")

BASE = "https://storage.googleapis.com/isi-surgvu"
URLS: dict[str, str] = {
    "videos": f"{BASE}/surgvu24_videos_only.zip",
    "labels": f"{BASE}/surgvu24_labels_updated_v2.zip",
    "validation": f"{BASE}/cat1_test_set_public.zip",
}
DESCRIPTIONS = {
    "videos": "280 clips, 155 sessions, 720p60, ~840 hours (very large)",
    "labels": "tools.csv + tasks.csv interval annotations (small)",
    "validation": "Category-1 public test set",
}


def remote_size(url: str) -> Optional[int]:
    try:
        import httpx

        with httpx.Client(timeout=15, follow_redirects=True) as client:
            response = client.head(url)
            if response.status_code == 200:
                return int(response.headers.get("content-length", 0)) or None
    except Exception:
        pass

    if shutil.which("curl"):
        try:
            result = subprocess.run(
                ["curl", "-sIL", url], capture_output=True, text=True, timeout=20
            )
            for line in result.stdout.splitlines():
                if line.lower().startswith("content-length:"):
                    return int(line.split(":", 1)[1].strip())
        except (subprocess.SubprocessError, ValueError):
            pass
    return None


def human(size: Optional[int]) -> str:
    if not size:
        return "unknown"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"


def download(url: str, destination: Path) -> Optional[Path]:
    """Download with resume support, preferring curl, then httpx."""
    filename = url.rsplit("/", 1)[-1]
    target = destination / filename

    expected = remote_size(url)
    if target.exists() and expected and target.stat().st_size == expected:
        logger.info("%s already complete (%s) — skipping.", filename, human(expected))
        return target

    logger.info("Downloading %s (%s) → %s", filename, human(expected), target)

    if shutil.which("curl"):
        # -C - resumes; --retry survives transient network failures.
        result = subprocess.run(
            ["curl", "-L", "-C", "-", "--retry", "3", "--progress-bar", "-o", str(target), url]
        )
        if result.returncode == 0:
            return target
        logger.warning("curl exited %d — trying httpx.", result.returncode)

    try:
        import httpx

        resume_from = target.stat().st_size if target.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        mode = "ab" if resume_from else "wb"

        with httpx.stream("GET", url, headers=headers, timeout=None, follow_redirects=True) as response:
            if response.status_code not in (200, 206):
                logger.error("HTTP %d for %s", response.status_code, url)
                return None
            written = resume_from
            with target.open(mode) as handle:
                for chunk in response.iter_bytes(1024 * 1024):
                    handle.write(chunk)
                    written += len(chunk)
                    if expected:
                        print(
                            f"\r  {human(written)} / {human(expected)} "
                            f"({100 * written / expected:.1f}%)",
                            end="", flush=True,
                        )
            print()
        return target
    except ImportError:
        logger.error("Neither curl nor httpx is available. Install: pip install httpx")
    except Exception as exc:
        logger.error("Download failed: %s", exc)
    return None


def extract(archive: Path, destination: Path) -> bool:
    logger.info("Extracting %s…", archive.name)
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(destination)
        logger.info("Extracted → %s", destination)
        return True
    except zipfile.BadZipFile:
        logger.error(
            "%s is not a valid zip — the download was probably truncated. "
            "Delete it and re-run.",
            archive.name,
        )
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the SurgVU dataset.")
    parser.add_argument("--dest", default="data/surgvu")
    parser.add_argument("--only", nargs="+", choices=list(URLS), help="Subset to fetch.")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--check", action="store_true", help="Report sizes without downloading.")
    parser.add_argument("--keep-archives", action="store_true",
                        help="Keep the .zip files after extraction.")
    args = parser.parse_args()

    wanted = args.only or list(URLS)

    if args.check:
        print("\nSurgVU dataset (arXiv:2501.09209v1)\n" + "-" * 62)
        for key in wanted:
            print(f"  {key:<12} {human(remote_size(URLS[key])):>12}   {DESCRIPTIONS[key]}")
        print()
        return 0

    destination = Path(args.dest)
    destination.mkdir(parents=True, exist_ok=True)

    free = shutil.disk_usage(destination).free
    logger.info("Destination: %s (%s free)", destination.resolve(), human(free))
    if "videos" in wanted and free < 150 * 1024**3:
        logger.warning(
            "Less than 150 GB free. The video archive needs roughly 100 GB for "
            "the zip plus the same again once extracted."
        )

    failed = []
    for key in wanted:
        archive = download(URLS[key], destination)
        if archive is None:
            failed.append(key)
            continue
        if args.extract:
            if extract(archive, destination) and not args.keep_archives:
                archive.unlink(missing_ok=True)
                logger.info("Removed %s (use --keep-archives to keep it)", archive.name)

    if failed:
        logger.error("Failed: %s", ", ".join(failed))
        return 1

    print(
        "\nDone. Next steps:\n"
        f"  1. python -m ai.training.extract_frames --dataset {destination} --fps 1\n"
        "  2. python -m ai.training.train_tool_detection\n"
        "  3. python -m ai.training.train_step_recognition --init-from checkpoints/tool_best.pt\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
