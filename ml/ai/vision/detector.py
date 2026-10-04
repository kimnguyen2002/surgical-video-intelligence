"""
Serving-side instrument detector.

Loads the checkpoint produced by :mod:`ai.training.train_detection` and turns a
frame into labelled boxes. This is the *predicted* counterpart to
:mod:`ai.datasets.groundtruth`, and it is deliberately built so the two can
never be confused downstream: every box this module emits carries
``provenance: "predicted"`` and a confidence, and the loader refuses to invent
output when there is no checkpoint.

That refusal is the important part. The prototype this replaced returned
hardcoded detections whenever inference failed, which on an explainability
platform is the worst possible failure mode: a fabricated 0.93 is
indistinguishable from a real one, and it propagates into the timeline, the
search index, the exported annotations, and every answer the assistant gives.
Here, no checkpoint means ``available == False`` and an empty list.

Ultralytics is imported lazily so an install without it still runs the platform
and simply reports the detector as unavailable.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ai.common import settings
from ai.datasets.coco import DETECTION_CLASSES, display_name

logger = logging.getLogger("charlie.vision.detector")

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Searched in order. The stable name is written by the trainer after each run
#: so the serving layer never has to know run directory names.
CHECKPOINT_CANDIDATES = (
    REPO_ROOT / "checkpoints" / "detection_best.pt",
    REPO_ROOT / "checkpoints" / "detection" / "cat1_detect_v2" / "weights" / "best.pt",
    REPO_ROOT / "checkpoints" / "detection" / "cat1_detect" / "weights" / "best.pt",
)


@dataclass
class DetectorStatus:
    available: bool = False
    checkpoint: Optional[str] = None
    device: str = "cpu"
    classes: list[str] = field(default_factory=lambda: list(DETECTION_CLASSES))
    reason: Optional[str] = None
    last_latency_ms: Optional[float] = None
    #: What this model was actually trained on. Carried into the UI so a
    #: small-data demonstrator is never presented as a validated detector.
    provenance_note: str = (
        "Trained on five clips of the public SurgVU cat1 set and validated on "
        "two held-out clips. Six of the fourteen classes never occur in that "
        "data and cannot be predicted. Educational and research use only."
    )

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "classes": self.classes,
            "reason": self.reason,
            "last_latency_ms": self.last_latency_ms,
            "provenance_note": self.provenance_note,
        }


def _resolve_device() -> str:
    requested = (settings.inference_device or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class InstrumentDetector:
    """Thread-safe lazy wrapper around the trained YOLO checkpoint."""

    def __init__(self) -> None:
        self._model: Any = None
        self._lock = threading.Lock()
        self._loaded = False
        self.status = DetectorStatus()

    # -- loading ----------------------------------------------------------

    def _find_checkpoint(self) -> Optional[Path]:
        override = (settings.__dict__.get("detector_checkpoint") or "").strip()
        if override:
            path = Path(override)
            return path if path.exists() else None
        for candidate in CHECKPOINT_CANDIDATES:
            if candidate.exists():
                return candidate
        return None

    def load(self, force: bool = False) -> DetectorStatus:
        with self._lock:
            if self._loaded and not force:
                return self.status

            self._model = None
            self._loaded = True
            self.status = DetectorStatus(device=_resolve_device())

            checkpoint = self._find_checkpoint()
            if checkpoint is None:
                self.status.reason = (
                    "No detection checkpoint. Train one with:  "
                    "python -m ai.training.export_detection && "
                    "python -m ai.training.train_detection"
                )
                logger.info("Instrument detector: %s", self.status.reason)
                return self.status

            try:
                from ultralytics import YOLO  # noqa: PLC0415
            except ImportError:
                self.status.reason = (
                    "Checkpoint found but Ultralytics is not installed "
                    "(pip install ultralytics)."
                )
                logger.warning("Instrument detector: %s", self.status.reason)
                return self.status

            try:
                self._model = YOLO(str(checkpoint))
                self.status.available = True
                self.status.checkpoint = str(checkpoint)
                # Prefer the checkpoint's own class names: if a model was
                # trained on a different vocabulary, using ours would relabel
                # every box.
                names = getattr(self._model, "names", None)
                if isinstance(names, dict) and names:
                    self.status.classes = [names[i] for i in sorted(names)]
                logger.info("Instrument detector loaded from %s", checkpoint.name)
            except Exception as exc:
                self.status.reason = f"Checkpoint failed to load: {exc}"
                logger.error("Instrument detector: %s", self.status.reason)

            return self.status

    def reload(self) -> dict:
        return self.load(force=True).to_dict()

    @property
    def available(self) -> bool:
        if not self._loaded:
            self.load()
        return self.status.available

    # -- inference --------------------------------------------------------

    def detect(
        self,
        image: Any,
        *,
        conf: float = 0.25,
        iou: float = 0.5,
        max_det: int = 20,
    ) -> dict:
        """
        Detect instruments in one frame.

        ``image`` may be anything Ultralytics accepts — a path, a PIL image, or
        a BGR numpy array. Boxes come back in *source-frame* pixels with the
        frame dimensions attached, matching the ground-truth payload shape so
        the client renders both with the same code.
        """
        if not self.available:
            return {
                "available": False,
                "boxes": [],
                "provenance": "none",
                "reason": self.status.reason,
            }

        started = time.perf_counter()
        try:
            results = self._model.predict(
                image, conf=conf, iou=iou, max_det=max_det, verbose=False,
                device=self.status.device,
            )
        except Exception as exc:
            logger.error("Detection failed: %s", exc)
            return {
                "available": True,
                "boxes": [],
                "provenance": "none",
                "error": str(exc),
            }

        latency = (time.perf_counter() - started) * 1000
        self.status.last_latency_ms = round(latency, 1)

        boxes: list[dict] = []
        if results:
            result = results[0]
            height, width = (getattr(result, "orig_shape", None) or (0, 0))[:2]
            for box in getattr(result, "boxes", []) or []:
                try:
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                    class_id = int(box.cls[0])
                    score = float(box.conf[0])
                except (AttributeError, IndexError, TypeError):
                    continue
                name = (
                    self.status.classes[class_id]
                    if 0 <= class_id < len(self.status.classes)
                    else f"class_{class_id}"
                )
                boxes.append(
                    {
                        "class": name,
                        "display": display_name(name),
                        "x": round(x1, 1),
                        "y": round(y1, 1),
                        "w": round(x2 - x1, 1),
                        "h": round(y2 - y1, 1),
                        "confidence": round(score, 4),
                        "frame_width": int(width),
                        "frame_height": int(height),
                    }
                )

        boxes.sort(key=lambda b: -b["confidence"])
        return {
            "available": True,
            "boxes": boxes,
            # Never "ground_truth". These are model output, and the whole
            # provenance system exists so that distinction survives to the UI.
            "provenance": "predicted",
            "latency_ms": round(latency, 1),
            "checkpoint": self.status.checkpoint,
            "note": self.status.provenance_note,
        }

    def detect_bytes(self, data: bytes, **kwargs) -> dict:
        """Detect on raw encoded image bytes (PNG/JPEG)."""
        try:
            import io  # noqa: PLC0415

            import numpy as np  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
        except ImportError as exc:
            return {"available": False, "boxes": [], "provenance": "none",
                    "reason": f"Pillow/numpy required: {exc}"}

        try:
            pil = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            return {"available": False, "boxes": [], "provenance": "none",
                    "reason": f"Unreadable image: {exc}"}

        # Ultralytics expects BGR when handed an array.
        array = np.asarray(pil)[:, :, ::-1]
        return self.detect(array, **kwargs)


detector = InstrumentDetector()
