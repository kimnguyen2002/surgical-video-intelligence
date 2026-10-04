"""
Discovery of the SurgVU corpora that are already on disk.

Search order for each corpus is: explicit environment variable, then a set of
conventional locations relative to the repository. Nothing is downloaded and
nothing is copied — a corpus is referenced where it lies, because the video set
is ~170 GB and duplicating it into ``data/`` is not an option.

Everything here is filesystem metadata only. Decoding, labelling, and training
live in :mod:`ai.training`; this module just answers "what is available, and
where".
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("charlie.datasets")

REPO_ROOT = Path(__file__).resolve().parents[2]

VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".mov", ".avi"}

#: ``case_051_video_part_002.mp4`` → case ``051``, part ``2``.
_PART_RE = re.compile(r"case[_\-]?(\d+).*?part[_\-]?(\d+)", re.IGNORECASE)
#: ``3_fps1.mp4`` → cat1 video ``3``.
_CAT1_RE = re.compile(r"^(\d+)_fps(\d+)$", re.IGNORECASE)
_DIGITS_RE = re.compile(r"(\d+)")


def _candidates(env_var: str, *relative: str) -> list[Path]:
    """Explicit override first, then conventional locations."""
    found: list[Path] = []
    override = os.environ.get(env_var, "").strip()
    if override:
        found.append(Path(override).expanduser())
    for rel in relative:
        # Siblings of the repo (``d:/surg/surge`` → ``d:/surg/<rel>``) come
        # first: that is where the released archives unpack in practice.
        found.append(REPO_ROOT.parent / rel)
        found.append(REPO_ROOT / rel)
        found.append(REPO_ROOT / "data" / rel)
    return found


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        try:
            if path.is_dir():
                return path.resolve()
        except OSError:  # pragma: no cover - unreadable mount
            continue
    return None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoAsset:
    """
    One playable video, however it got here.

    This is the single record the library, the player, and the chat assistant
    all speak. ``asset_id`` is stable across restarts and safe in a URL, so a
    deep link to a moment in a procedure keeps working.
    """

    asset_id: str
    title: str
    path: Path
    corpus: str
    case: str = ""
    part: int = 0
    size_bytes: int = 0
    duration_s: Optional[float] = None
    #: What ground truth exists for this video, e.g. ``{"tools", "tasks"}``.
    annotations: frozenset[str] = frozenset()

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / 1024**2, 1)

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "title": self.title,
            "corpus": self.corpus,
            "case": self.case,
            "part": self.part,
            "size_mb": self.size_mb,
            "duration_s": self.duration_s,
            "annotations": sorted(self.annotations),
            "stream_url": f"/api/library/stream/{self.asset_id}",
            "path": str(self.path),
        }


@dataclass
class SurgVUCase:
    """A surgvu24 case: one or more video parts plus its two label CSVs."""

    case: str
    parts: dict[int, Path] = field(default_factory=dict)
    tools_csv: Optional[Path] = None
    tasks_csv: Optional[Path] = None

    @property
    def has_labels(self) -> bool:
        return self.tools_csv is not None or self.tasks_csv is not None

    @property
    def has_video(self) -> bool:
        return bool(self.parts)

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.parts.values() if p.exists())


@dataclass(frozen=True)
class Cat1Video:
    """A cat1 clip with COCO bounding boxes (and the grand-challenge mirror)."""

    name: str
    video: Path
    coco: Path
    gc: Optional[Path] = None


@dataclass(frozen=True)
class Cat2Case:
    """A cat2 case: a clip, a question, and the accepted answers."""

    case: str
    video: Path
    question: str
    answers: tuple[str, ...]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class CorpusRegistry:
    """
    Lazily-resolved view of every corpus on this machine.

    Scanning 170 GB of directory entries is cheap but not free, so results are
    memoised. Call :meth:`refresh` after adding data.
    """

    def __init__(self) -> None:
        self._cache: dict[str, object] = {}

    # -- roots ------------------------------------------------------------

    @property
    def surgvu_video_root(self) -> Optional[Path]:
        return _first_existing(
            _candidates("SURGVU_VIDEO_ROOT", "surgvu24_videos_only", "surgvu24", "surgvu")
        )

    @property
    def surgvu_label_root(self) -> Optional[Path]:
        root = _first_existing(
            _candidates(
                "SURGVU_LABEL_ROOT",
                "surgvu24_labels_updated_v2",
                "surgvu24_labels",
            )
        )
        # The archive nests the cases one level down, under ``labels/``.
        if root is not None and (root / "labels").is_dir():
            return root / "labels"
        return root

    @property
    def cat1_root(self) -> Optional[Path]:
        return _first_existing(_candidates("SURGVU_CAT1_ROOT", "cat1_test_set_public"))

    @property
    def cat2_root(self) -> Optional[Path]:
        return _first_existing(
            _candidates("SURGVU_CAT2_ROOT", "SURGVU25_cat_2_sample_set_public")
        )

    # -- surgvu24 ---------------------------------------------------------

    @functools.cached_property
    def surgvu_cases(self) -> dict[str, SurgVUCase]:
        """
        Every surgvu24 case, keyed by zero-padded case id.

        Videos and labels are discovered independently and then joined, because
        they ship as separate archives and either can be absent: 155 cases have
        labels while only some have video downloaded. A case with labels but no
        video is still listed — that is what tells you what is missing.
        """
        cases: dict[str, SurgVUCase] = {}

        video_root = self.surgvu_video_root
        if video_root is not None:
            # ``rglob`` covers both released layouts: ``case_051/*.mp4`` at the
            # top level and ``surgvu24/case_000/*.mp4`` nested one deeper.
            for path in video_root.rglob("*.mp4"):
                if ".ipynb_checkpoints" in path.parts:
                    continue
                match = _PART_RE.search(path.stem)
                if match:
                    case, part = match.group(1), int(match.group(2))
                else:
                    digits = _DIGITS_RE.findall(path.parent.name)
                    case, part = (digits[-1] if digits else path.parent.name), 1
                case = case.zfill(3)
                cases.setdefault(case, SurgVUCase(case=case)).parts[part] = path

        label_root = self.surgvu_label_root
        if label_root is not None:
            for case_dir in label_root.iterdir():
                if not case_dir.is_dir() or case_dir.name.startswith("."):
                    continue
                digits = _DIGITS_RE.findall(case_dir.name)
                case = (digits[-1] if digits else case_dir.name).zfill(3)
                record = cases.setdefault(case, SurgVUCase(case=case))
                tools, tasks = case_dir / "tools.csv", case_dir / "tasks.csv"
                if tools.exists():
                    record.tools_csv = tools
                if tasks.exists():
                    record.tasks_csv = tasks

        return dict(sorted(cases.items()))

    # -- cat1 -------------------------------------------------------------

    @functools.cached_property
    def cat1_videos(self) -> dict[str, Cat1Video]:
        """cat1 clips that have both a video and COCO annotations."""
        root = self.cat1_root
        if root is None:
            return {}

        videos: dict[str, Cat1Video] = {}
        for path in sorted(root.glob("*.mp4")):
            if "__MACOSX" in path.parts:
                continue
            coco = path.with_name(f"{path.stem}_coco.json")
            if not coco.exists():
                logger.debug("cat1 clip %s has no COCO file; skipping", path.name)
                continue
            gc = path.with_name(f"{path.stem}_gc.json")
            match = _CAT1_RE.match(path.stem)
            name = match.group(1) if match else path.stem
            videos[name] = Cat1Video(
                name=name, video=path, coco=coco, gc=gc if gc.exists() else None
            )
        return dict(sorted(videos.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0))

    # -- cat2 -------------------------------------------------------------

    @functools.cached_property
    def cat2_cases(self) -> dict[str, Cat2Case]:
        """
        cat2 VQA cases.

        Each case directory holds ``<case>.mp4``, ``<case>_question.json``
        (a single question string) and ``<case>.json`` (a list of accepted
        free-text answers — the task is open-ended, so there are several
        phrasings of the same correct answer).
        """
        root = self.cat2_root
        if root is None:
            return {}

        cases: dict[str, Cat2Case] = {}
        for case_dir in sorted(root.iterdir()):
            if not case_dir.is_dir() or case_dir.name.startswith("."):
                continue
            name = case_dir.name
            video = case_dir / f"{name}.mp4"
            q_path = case_dir / f"{name}_question.json"
            a_path = case_dir / f"{name}.json"
            if not (video.exists() and q_path.exists() and a_path.exists()):
                continue
            try:
                question = json.loads(q_path.read_text(encoding="utf-8"))
                answers = json.loads(a_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("cat2 case %s is unreadable: %s", name, exc)
                continue
            if isinstance(answers, str):
                answers = [answers]
            cases[name] = Cat2Case(
                case=name,
                video=video,
                question=str(question),
                answers=tuple(str(a) for a in answers),
            )
        return cases

    # -- unified library --------------------------------------------------

    def assets(self) -> list[VideoAsset]:
        """
        Every discovered video as a uniform, playable record.

        This is what the library UI lists and what the assistant grounds its
        answers in — local footage with real annotations, rather than a YouTube
        URL whose content nothing in the system can verify.
        """
        out: list[VideoAsset] = []

        for case in self.surgvu_cases.values():
            annotations = set()
            if case.tools_csv:
                annotations.add("tools")
            if case.tasks_csv:
                annotations.add("tasks")
            for part, path in sorted(case.parts.items()):
                out.append(
                    VideoAsset(
                        asset_id=f"surgvu24-{case.case}-{part:03d}",
                        title=f"SurgVU case {case.case} · part {part}",
                        path=path,
                        corpus="surgvu24",
                        case=case.case,
                        part=part,
                        size_bytes=_size(path),
                        annotations=frozenset(annotations),
                    )
                )

        for name, clip in self.cat1_videos.items():
            out.append(
                VideoAsset(
                    asset_id=f"cat1-{name}",
                    title=f"SurgVU cat1 clip {name} · boxed",
                    path=clip.video,
                    corpus="cat1",
                    case=name,
                    size_bytes=_size(clip.video),
                    annotations=frozenset({"boxes"}),
                )
            )

        for name, case in self.cat2_cases.items():
            out.append(
                VideoAsset(
                    asset_id=f"cat2-{name}",
                    title=f"SurgVU cat2 {name} · VQA",
                    path=case.video,
                    corpus="cat2",
                    case=name,
                    size_bytes=_size(case.video),
                    annotations=frozenset({"vqa"}),
                )
            )

        return out

    def asset(self, asset_id: str) -> Optional[VideoAsset]:
        for item in self.assets():
            if item.asset_id == asset_id:
                return item
        return None

    # -- reporting --------------------------------------------------------

    def summary(self) -> dict:
        """A short, honest description of what is and is not present."""
        cases = self.surgvu_cases
        with_video = [c for c in cases.values() if c.has_video]
        with_labels = [c for c in cases.values() if c.has_labels]
        both = [c for c in cases.values() if c.has_video and c.has_labels]
        total_bytes = sum(c.total_bytes() for c in with_video)

        return {
            "surgvu24": {
                "root": str(self.surgvu_video_root) if self.surgvu_video_root else None,
                "labels_root": (
                    str(self.surgvu_label_root) if self.surgvu_label_root else None
                ),
                "cases_total": len(cases),
                "cases_with_video": len(with_video),
                "cases_with_labels": len(with_labels),
                "cases_trainable": len(both),
                "video_parts": sum(len(c.parts) for c in with_video),
                "video_gb": round(total_bytes / 1024**3, 1),
            },
            "cat1": {
                "root": str(self.cat1_root) if self.cat1_root else None,
                "clips": len(self.cat1_videos),
                "annotation": "COCO bounding boxes",
            },
            "cat2": {
                "root": str(self.cat2_root) if self.cat2_root else None,
                "cases": len(self.cat2_cases),
                "annotation": "open-ended question / answer pairs",
            },
        }

    def refresh(self) -> None:
        """Drop memoised scans — call after adding or removing data."""
        for attr in ("surgvu_cases", "cat1_videos", "cat2_cases"):
            self.__dict__.pop(attr, None)

    def __iter__(self) -> Iterator[VideoAsset]:
        return iter(self.assets())


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


registry = CorpusRegistry()
