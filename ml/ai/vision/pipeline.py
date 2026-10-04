"""
Live perception pipeline.

:class:`FrameAnalyzer` analyses one frame: instrument presence, surgical
phase, a Grad-CAM explanation, and a feature embedding.

:class:`VideoIndexer` walks a whole video, feeds the observations to the
temporal memory engine, and produces the searchable timeline.

**On honesty about predictions.** When no trained checkpoint is present, this
module does *not* invent plausible-looking tool detections. It returns real
embeddings (which need no task supervision) and reports
``predictions_available: false`` with the command that would train a model.
Fabricated confidence scores in a platform whose entire premise is
explainability would be worse than no scores at all — a user cannot tell a
made-up 0.93 from a real one, and every downstream timeline, search result,
and explanation would inherit the fiction.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ai.common import optional_import, settings
from ai.embeddings import get_frame_embedder
from ai.knowledge import knowledge_graph
from ai.temporal_memory import FrameObservation, memory_engine

logger = logging.getLogger("charlie.vision")


@dataclass
class AnalyzerStatus:
    tool_model: Optional[str] = None
    step_model: Optional[str] = None
    task_model: Optional[str] = None
    device: str = "cpu"
    predictions_available: bool = False
    reason: str = ""
    frames_processed: int = 0
    total_inference_ms: float = 0.0
    embedder: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool_model": self.tool_model,
            "step_model": self.step_model,
            "task_model": self.task_model,
            "device": self.device,
            "predictions_available": self.predictions_available,
            "reason": self.reason,
            "frames_processed": self.frames_processed,
            "avg_inference_ms": (
                round(self.total_inference_ms / self.frames_processed, 2)
                if self.frames_processed
                else 0.0
            ),
            "embedder": self.embedder,
        }


class FrameAnalyzer:
    """Analyses individual frames with whatever models are actually available."""

    def __init__(self):
        self._lock = threading.RLock()
        self._tool_model = None
        self._step_model = None
        self._task_model = None
        self._tool_classes: list[str] = []
        self._task_classes: list[str] = []
        self._task_meta: dict = {}
        self._transform = None
        self.status = AnalyzerStatus(embedder=get_frame_embedder().info())
        self._load()

    # -- loading -----------------------------------------------------------
    def _load(self) -> None:
        torch = optional_import("torch")
        if torch is None:
            self.status.reason = (
                "PyTorch is not installed, so no vision model can run. "
                "Install it with: pip install torch torchvision timm"
            )
            logger.info(self.status.reason)
            return

        from ai.embeddings.vision import resolve_device

        self.status.device = resolve_device(settings.inference_device)

        checkpoint = self._resolve_checkpoint(settings.tool_checkpoint, "tool_best.pt")
        if checkpoint is None:
            self.status.reason = (
                "No trained SurgVU checkpoint found. Frame embeddings, timeline "
                "segmentation, and semantic search work; instrument and phase "
                "predictions need a trained model. Train one with: "
                "python -m ai.training.train_tool_detection"
            )
            logger.info(self.status.reason)
            return

        try:
            from ai.training.config import config
            from ai.training.dataset import default_transform
            from ai.training.models import load_checkpoint

            model, state = load_checkpoint(checkpoint, task="tool", device=self.status.device)
            self._tool_model = model
            self._tool_classes = state.get("classes", config.tool_classes)
            self._transform = default_transform(train=False)
            self.status.tool_model = f"{state.get('model_name', 'tool')}@{checkpoint.name}"
            self.status.predictions_available = True
            logger.info("Loaded tool detector: %s", self.status.tool_model)
        except Exception as exc:
            self.status.reason = f"Could not load {checkpoint}: {exc}"
            logger.error(self.status.reason)
            return

        # Surgical task recogniser — what is being *done*, from tasks.csv.
        #
        # A separate slot from the step recogniser: that one is a temporal clip
        # model, this is a single-frame classifier of the same architecture as
        # the tool head, which means it also exposes a Grad-CAM layer. It is
        # the model that answers "the surgeon is suturing" rather than merely
        # "a needle driver is mounted".
        task_checkpoint = self._resolve_checkpoint("", "task_best.pt")
        if task_checkpoint is not None:
            try:
                from ai.training.models import load_checkpoint

                model, state = load_checkpoint(
                    task_checkpoint, task="tool", device=self.status.device
                )
                self._task_model = model
                self._task_classes = state.get("classes", [])
                self._task_meta = {
                    "balanced_accuracy": state.get("val_balanced_accuracy"),
                    "epoch": state.get("epoch"),
                    "note": state.get("provenance_note"),
                }
                self.status.task_model = (
                    f"{state.get('model_name', 'task')}@{task_checkpoint.name}"
                )
                logger.info("Loaded task recogniser: %s", self.status.task_model)
            except Exception as exc:
                logger.warning("Could not load task checkpoint: %s", exc)

        step_checkpoint = self._resolve_checkpoint(settings.step_checkpoint, "step_best.pt")
        if step_checkpoint is not None:
            try:
                from ai.training.models import load_checkpoint

                model, state = load_checkpoint(
                    step_checkpoint, task="step", device=self.status.device
                )
                self._step_model = model
                self._step_classes = state.get("classes", [])
                self.status.step_model = f"{state.get('model_name', 'step')}@{step_checkpoint.name}"
                logger.info("Loaded step recogniser: %s", self.status.step_model)
            except Exception as exc:
                logger.warning("Could not load step checkpoint: %s", exc)

    @staticmethod
    def _resolve_checkpoint(configured: str, default_name: str) -> Optional[Path]:
        if configured:
            path = Path(configured)
            return path if path.exists() else None
        from ai.training.config import config

        candidate = Path(config.output_dir) / default_name
        return candidate if candidate.exists() else None

    def reload(self) -> dict:
        """Re-scan for checkpoints — lets a fresh training run go live."""
        with self._lock:
            self._tool_model = None
            self._step_model = None
            self.status = AnalyzerStatus(embedder=get_frame_embedder().info())
            self._load()
        return self.status.to_dict()

    # -- analysis ----------------------------------------------------------
    def analyze(
        self,
        image,
        specialty: str = "General Surgery",
        want_gradcam: bool = True,
        threshold: Optional[float] = None,
    ) -> dict:
        """
        Analyse one PIL image.

        Always returns an embedding. Returns tool/phase predictions and a
        Grad-CAM overlay only when a trained model is loaded.
        """
        started = time.time()
        embedder = get_frame_embedder()
        embedding = embedder.encode_pil([image])[0]

        result: dict[str, Any] = {
            "embedding_dim": len(embedding),
            "embedding": embedding,
            "predictions_available": self.status.predictions_available,
            "model_name": self.status.tool_model or embedder.backend,
            "device": self.status.device,
            "specialty": specialty,
            "disclaimer": settings.disclaimer,
        }

        if not self.status.predictions_available:
            result["notice"] = self.status.reason
            result["inference_ms"] = round((time.time() - started) * 1000, 2)
            return result

        # Out-of-domain gate.
        #
        # The classifier was trained only on endoscopic frames and has no way
        # to express "I have never seen anything like this" — point a webcam at
        # a face and it reports a needle driver at 0.98, with a Grad-CAM
        # heatmap over the person's cheek. Withholding predictions on input the
        # model has no competence over is the difference between a tool that
        # can be trusted and one that merely looks confident.
        from ai.vision.domain_guard import domain_guard  # noqa: PLC0415

        verdict = domain_guard.check(embedding, embedder=embedder.backend)
        result["domain"] = verdict.to_dict()
        if not verdict.in_domain:
            result["predictions_available"] = False
            result["out_of_domain"] = True
            result["notice"] = verdict.reason
            result["tools"] = []
            result["inference_ms"] = round((time.time() - started) * 1000, 2)
            return result

        import torch

        from ai.training.config import config

        threshold = config.tool_threshold if threshold is None else threshold

        with self._lock:
            tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self.status.device)
            with torch.no_grad():
                probabilities = torch.sigmoid(self._tool_model(tensor))[0].float().cpu().tolist()

            tools = [
                {
                    "name": name,
                    "display": config.tool_display(name),
                    "confidence": round(probability, 4),
                    "present": probability >= threshold,
                }
                for name, probability in zip(self._tool_classes, probabilities)
            ]
            tools.sort(key=lambda t: -t["confidence"])
            result["tools"] = tools
            result["threshold"] = threshold

            if want_gradcam and settings.enable_gradcam:
                result["gradcam_data_url"], result["gradcam_class"] = self._gradcam(
                    tensor, image, tools
                )

            self.status.frames_processed += 1

        # A trained task recogniser is strictly better evidence than guessing
        # the phase from which instruments are mounted, so it wins when loaded.
        result["phase"] = self._predict_task(image) or self._infer_phase(
            result.get("tools", []), specialty
        )
        result["inference_ms"] = round((time.time() - started) * 1000, 2)
        self.status.total_inference_ms += result["inference_ms"]
        return result

    def _gradcam(self, tensor, image, tools: list[dict]) -> tuple[Optional[str], Optional[str]]:
        """Explain the single most confident instrument prediction."""
        try:
            from ai.explainability import GradCAM, heatmap_to_png

            layer = self._tool_model.gradcam_layer()
            if layer is None:
                return None, None
            top = tools[0]["name"]
            class_index = self._tool_classes.index(top)
            with GradCAM(self._tool_model, layer) as cam:
                heatmap = cam.generate(tensor, class_idx=class_index)
            if heatmap is None:
                return None, None
            return heatmap_to_png(image.convert("RGB"), heatmap, alpha=0.55), top
        except Exception as exc:
            logger.warning("Grad-CAM failed: %s", exc)
            return None, None

    def _predict_task(self, image) -> Optional[dict]:
        """
        Predict the surgical task from the trained `tasks.csv` recogniser.

        Single-label softmax, so the probabilities are a real distribution and
        the runner-up is meaningful — it is returned alongside, because a task
        classifier that is torn between suturing and retraction is telling you
        something a bare top-1 label hides.
        """
        if self._task_model is None or not self._task_classes:
            return None

        import torch

        from ai.training.config import TASK_DISPLAY

        try:
            with self._lock:
                tensor = (
                    self._transform(image.convert("RGB"))
                    .unsqueeze(0)
                    .to(self.status.device)
                )
                with torch.no_grad():
                    probabilities = (
                        torch.softmax(self._task_model(tensor), dim=1)[0]
                        .float()
                        .cpu()
                        .tolist()
                    )
        except Exception as exc:
            logger.warning("Task prediction failed: %s", exc)
            return None

        ranked = sorted(
            zip(self._task_classes, probabilities), key=lambda pair: -pair[1]
        )
        name, confidence = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None

        return {
            "phase": TASK_DISPLAY.get(name, name.replace("_", " ").title()),
            "raw": name,
            "confidence": round(float(confidence), 3),
            "source": "trained task recogniser (SurgVU tasks.csv)",
            "explanation": (
                f"A classifier trained on SurgVU task intervals scores this "
                f"frame as {TASK_DISPLAY.get(name, name)} "
                f"({confidence:.0%})."
                + (
                    f" Next most likely: {TASK_DISPLAY.get(runner_up[0], runner_up[0])} "
                    f"({runner_up[1]:.0%})."
                    if runner_up and runner_up[1] > 0.15
                    else ""
                )
            ),
            "alternatives": [
                {
                    "phase": TASK_DISPLAY.get(n, n),
                    "confidence": round(float(p), 3),
                }
                for n, p in ranked[1:4]
                if p > 0.05
            ],
            "next_phase": None,
            "progress": None,
            "model": self.status.task_model,
        }

    def _infer_phase(self, tools: list[dict], specialty: str) -> Optional[dict]:
        """
        Estimate the phase from detected instruments and the knowledge graph.

        This is explicitly an *inference from instruments*, not a trained phase
        model, and it is labelled as such in the response so the UI never
        presents it as a direct observation. When a trained step recogniser is
        loaded, that takes over — it is strictly better evidence.
        """
        present = [t["name"] for t in tools if t["present"]]
        if not present:
            return None

        phases = knowledge_graph.phases(specialty)
        if not phases:
            return None

        best_phase, best_score = None, 0.0
        for phase in phases:
            haystack = f"{phase['name']} {phase['description']}".lower()
            score = sum(
                1 for tool in present if any(word in haystack for word in tool.split("_"))
            )
            if score > best_score:
                best_phase, best_score = phase, score

        if best_phase is None:
            return None

        confidence = min(0.35 + 0.15 * best_score, 0.75)
        return {
            "phase": best_phase["name"],
            "confidence": round(confidence, 3),
            "source": "instrument-inferred",
            "explanation": (
                f"Inferred from the instruments detected ({', '.join(present[:3])}) "
                f"and the {specialty} workflow graph — not a direct phase-model "
                f"prediction. Train a step recogniser for a measured phase output."
            ),
            "next_phase": knowledge_graph.next_phase(specialty, best_phase["name"]),
            "progress": knowledge_graph.progress(specialty, best_phase["name"]),
        }

    def analyze_bytes(self, data: bytes, **kwargs) -> dict:
        pil = optional_import("pillow")
        if pil is None:
            return {
                "error": "Pillow is required to decode images. pip install Pillow",
                "predictions_available": False,
            }
        import io

        from PIL import Image

        try:
            image = Image.open(io.BytesIO(data))
        except Exception as exc:
            return {"error": f"Could not decode image: {exc}", "predictions_available": False}
        return self.analyze(image, **kwargs)


# ---------------------------------------------------------------------------
# Video indexing
# ---------------------------------------------------------------------------
class VideoIndexer:
    """Samples a video, analyses each frame, and builds its temporal memory."""

    def __init__(self, analyzer: Optional[FrameAnalyzer] = None):
        self.analyzer = analyzer or frame_analyzer
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def job_status(self, session_id: str) -> dict:
        return self._jobs.get(session_id, {"status": "unknown"})

    def index_video(
        self,
        video_path: str,
        session_id: str,
        specialty: str = "General Surgery",
        interval: Optional[float] = None,
        max_frames: int = 400,
        progress: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """
        Analyse a video end to end and store its temporal memory.

        Blocking — the API layer runs this on a background task.

        Two things keep this bounded, because the naive version took ~18
        minutes on a five-hour recording and the timeline is useless until it
        finishes:

        **Seek, do not walk.** Sampling every *n*-th frame by calling ``grab()``
        on all the others means demuxing about a million frames for a 5-hour
        60 fps video — measured at ~11 minutes, before any inference. Seeking
        straight to each wanted position reads three orders of magnitude fewer
        frames. Sequential reading is still used when the step is small, where
        seek overhead would dominate.

        **400 frames, not 2000.** The timeline is a navigation aid; 400 points
        across a procedure is one every 45 seconds on a five-hour case, which
        is finer than the phases being segmented. The extra 1600 frames cost
        five minutes of inference and changed nothing a viewer could see.
        """
        interval = interval or settings.memory_sample_interval
        cv2 = optional_import("cv2")
        if cv2 is None:
            message = (
                "Video indexing needs OpenCV. Install: pip install opencv-python-headless"
            )
            memory_engine.mark_error(session_id, message)
            return {"ok": False, "error": message}

        path = Path(video_path)
        if not path.exists():
            message = f"Video not found: {path}"
            memory_engine.mark_error(session_id, message)
            return {"ok": False, "error": message}

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            message = f"Could not open video: {path}"
            memory_engine.mark_error(session_id, message)
            return {"ok": False, "error": message}

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if total_frames else 0.0
        step = max(1, int(round(fps * interval)))

        # Keep long videos bounded: widen the sampling interval rather than
        # truncating, so the timeline still spans the whole procedure.
        if total_frames and (total_frames / step) > max_frames:
            step = max(step, total_frames // max_frames)

        with self._lock:
            self._jobs[session_id] = {
                "status": "indexing", "processed": 0, "total": total_frames // step or 1,
                "duration": duration,
            }

        from PIL import Image

        observations: list[FrameObservation] = []
        started = time.time()

        def record(frame, position: int) -> None:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            analysis = self.analyzer.analyze(
                image, specialty=specialty, want_gradcam=False
            )
            observations.append(
                FrameObservation(
                    timestamp=position / fps,
                    embedding=analysis.get("embedding", []),
                    phase=(analysis.get("phase") or {}).get("phase"),
                    phase_confidence=(analysis.get("phase") or {}).get("confidence", 0.0),
                    tools=[
                        t["name"] for t in analysis.get("tools", []) if t.get("present")
                    ],
                )
            )
            with self._lock:
                self._jobs[session_id]["processed"] = len(observations)
            if progress and len(observations) % 10 == 0:
                progress(dict(self._jobs[session_id]))

        # Below this stride, seeking costs more than it saves: each seek lands
        # on a keyframe and decodes forward, so for closely-spaced samples a
        # straight sequential read is cheaper.
        SEEK_THRESHOLD = 60

        try:
            if step >= SEEK_THRESHOLD and total_frames:
                for position in range(0, total_frames, step):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        continue
                    record(frame, position)
                    if len(observations) >= max_frames:
                        break
            else:
                index = 0
                while True:
                    if not capture.grab():
                        break
                    if index % step == 0:
                        ok, frame = capture.retrieve()
                        if ok:
                            record(frame, index)
                            if len(observations) >= max_frames:
                                break
                    index += 1
        finally:
            capture.release()

        if not observations:
            message = "No frames could be decoded from this video."
            memory_engine.mark_error(session_id, message)
            with self._lock:
                self._jobs[session_id] = {"status": "error", "error": message}
            return {"ok": False, "error": message}

        events = memory_engine.ingest_observations(
            session_id, observations, duration=duration or observations[-1].timestamp
        )

        result = {
            "ok": True,
            "session_id": session_id,
            "frames_analyzed": len(observations),
            "events": len(events),
            "duration": round(duration, 2),
            "elapsed_s": round(time.time() - started, 1),
            "predictions_available": self.analyzer.status.predictions_available,
        }
        with self._lock:
            self._jobs[session_id] = {"status": "complete", **result}
        logger.info(
            "Indexed %s: %d frames → %d events in %.1fs",
            path.name, len(observations), len(events), result["elapsed_s"],
        )
        return result


frame_analyzer = FrameAnalyzer()
video_indexer = VideoIndexer(frame_analyzer)
