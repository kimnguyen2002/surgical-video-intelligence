"""
Optional-dependency capability layer.

The platform must run in *any* environment — a laptop with nothing but the
standard library, a CPU-only Docker container, or an 8×A100 training node.
Rather than sprinkling ``try: import torch`` throughout the codebase, every
optional dependency is declared here once. Modules ask
``capabilities.has("torch")`` and pick an implementation accordingly, and the
developer console renders :func:`capability_report` so it is always obvious
which tier the running instance is on.
"""

from __future__ import annotations

import importlib
import logging
import shutil
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("charlie.capabilities")


@dataclass
class Capability:
    """A single optional dependency and what the platform does without it."""

    name: str
    module: Optional[str]
    purpose: str
    install: str
    #: Executable that satisfies the capability instead of a Python module.
    binary: Optional[str] = None
    #: What happens when the capability is missing.
    fallback: str = "A lighter pure-Python implementation is used."
    available: bool = field(default=False, init=False)
    version: Optional[str] = field(default=None, init=False)
    detail: Optional[str] = field(default=None, init=False)


CAPABILITIES: dict[str, Capability] = {
    c.name: c
    for c in [
        Capability(
            name="numpy",
            module="numpy",
            purpose="Vectorised similarity search and array maths",
            install="pip install numpy",
            fallback="Pure-Python list maths (correct, ~50× slower).",
        ),
        Capability(
            name="torch",
            module="torch",
            purpose="Neural network inference and training",
            install="pip install torch torchvision",
            fallback="Vision falls back to classical descriptors; training is unavailable.",
        ),
        Capability(
            name="timm",
            module="timm",
            purpose="ConvNeXt / ViT / DINOv2 backbones",
            install="pip install timm",
            fallback="Colour+edge histogram descriptors are used for frame embeddings.",
        ),
        Capability(
            name="cv2",
            module="cv2",
            purpose="Video decoding, frame extraction, heatmap rendering",
            install="pip install opencv-python-headless",
            fallback="Frame extraction delegates to the ffmpeg binary if present.",
        ),
        Capability(
            name="pillow",
            module="PIL",
            purpose="Image decoding and resizing",
            install="pip install Pillow",
            fallback="Frame analysis is disabled; the rest of the platform is unaffected.",
        ),
        Capability(
            name="faiss",
            module="faiss",
            purpose="Approximate nearest-neighbour vector search at scale",
            install="pip install faiss-cpu",
            fallback="Exact brute-force cosine search (fine below ~100k vectors).",
        ),
        Capability(
            name="sentence_transformers",
            module="sentence_transformers",
            purpose="Dense semantic text embeddings for RAG",
            install="pip install sentence-transformers",
            fallback="Deterministic hashed n-gram embeddings (lexical, fully offline).",
        ),
        Capability(
            name="faster_whisper",
            module="faster_whisper",
            purpose="Local speech-to-text",
            install="pip install faster-whisper",
            fallback="Browser Web Speech API handles transcription client-side.",
        ),
        Capability(
            name="whisper",
            module="whisper",
            purpose="Local speech-to-text (reference implementation)",
            install="pip install openai-whisper",
            fallback="faster-whisper or the browser Web Speech API is used instead.",
        ),
        Capability(
            name="piper",
            module=None,
            binary="piper",
            purpose="Local neural text-to-speech",
            install="pip install piper-tts  # plus a voice model",
            fallback="Browser SpeechSynthesis speaks the reply client-side.",
        ),
        Capability(
            name="ffmpeg",
            module=None,
            binary="ffmpeg",
            purpose="Video transcoding and frame extraction",
            install="brew install ffmpeg  |  apt-get install ffmpeg",
            fallback="OpenCV handles decoding when available.",
        ),
        Capability(
            name="yt_dlp",
            module="yt_dlp",
            purpose="Retrieving educational videos from YouTube",
            install="pip install yt-dlp",
            fallback="The yt-dlp CLI is used if it is on PATH.",
        ),
        Capability(
            name="pypdf",
            module="pypdf",
            purpose="PDF text extraction for the RAG corpus",
            install="pip install pypdf",
            fallback="PDFs are rejected with a clear message; text formats still work.",
        ),
        Capability(
            name="pptx",
            module="pptx",
            purpose="PowerPoint text extraction for the RAG corpus",
            install="pip install python-pptx",
            fallback="PPTX uploads are rejected with a clear message.",
        ),
        Capability(
            name="docx",
            module="docx",
            purpose="Word document text extraction for the RAG corpus",
            install="pip install python-docx",
            fallback="DOCX uploads are rejected with a clear message.",
        ),
        Capability(
            name="google_genai",
            module="google.genai",
            purpose="Gemini chat provider (optional cloud backend)",
            install="pip install google-genai",
            fallback="Ollama or the built-in knowledge engine answers instead.",
        ),
        Capability(
            name="prometheus_client",
            module="prometheus_client",
            purpose="Prometheus metrics endpoint",
            install="pip install prometheus-client",
            fallback="/api/system/metrics serves a hand-rolled text exposition.",
        ),
        Capability(
            name="sklearn",
            module="sklearn",
            purpose="Evaluation metrics (mAP, F1, confusion matrices)",
            install="pip install scikit-learn",
            fallback="Built-in metric implementations in ai/training/metrics.py.",
        ),
    ]
}

#: Cache of successfully imported optional modules.
_MODULE_CACHE: dict[str, Any] = {}


def _probe(cap: Capability) -> None:
    """Detect a single capability, recording version/detail on the record."""
    if cap.binary:
        path = shutil.which(cap.binary)
        cap.available = path is not None
        cap.detail = path
        return

    if not cap.module:
        cap.available = False
        return

    try:
        module = importlib.import_module(cap.module)
    except Exception as exc:  # ImportError, but also broken native builds
        cap.available = False
        cap.detail = str(exc)[:200]
        return

    cap.available = True
    _MODULE_CACHE[cap.name] = module
    cap.version = getattr(module, "__version__", None)


_PROBED = False


def _probe_all() -> None:
    global _PROBED
    if _PROBED:
        return
    for cap in CAPABILITIES.values():
        _probe(cap)
    _PROBED = True
    enabled = sorted(c.name for c in CAPABILITIES.values() if c.available)
    logger.info("Capabilities detected: %s", ", ".join(enabled) or "none (pure-Python tier)")


def has(name: str) -> bool:
    """Return True when the named optional dependency is usable."""
    _probe_all()
    cap = CAPABILITIES.get(name)
    return bool(cap and cap.available)


def optional_import(name: str) -> Optional[Any]:
    """
    Return the imported module for a capability, or ``None`` when unavailable.

    >>> np = optional_import("numpy")
    >>> if np is not None:
    ...     ...
    """
    _probe_all()
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]
    cap = CAPABILITIES.get(name)
    if cap is None or not cap.module or not cap.available:
        return None
    try:
        module = importlib.import_module(cap.module)
    except Exception:
        return None
    _MODULE_CACHE[name] = module
    return module


def capability_report() -> dict:
    """
    A JSON-serialisable summary of what this instance can do.

    Surfaced verbatim at ``GET /api/system/capabilities`` and rendered in the
    developer console so operators can see which tier they are running and
    exactly which command unlocks the next one.
    """
    _probe_all()
    caps = []
    for cap in CAPABILITIES.values():
        caps.append(
            {
                "name": cap.name,
                "available": cap.available,
                "version": cap.version,
                "purpose": cap.purpose,
                "install": cap.install,
                "fallback": None if cap.available else cap.fallback,
                "detail": cap.detail,
            }
        )

    tier = "pure-python"
    if has("torch") and has("timm"):
        tier = "full-inference"
    elif has("numpy"):
        tier = "accelerated"

    gpu = _gpu_report()

    return {
        "tier": tier,
        "capabilities": sorted(caps, key=lambda c: (not c["available"], c["name"])),
        "available_count": sum(1 for c in caps if c["available"]),
        "total_count": len(caps),
        "gpu": gpu,
    }


def _gpu_report() -> dict:
    """GPU availability and memory, when torch is installed."""
    torch = optional_import("torch")
    if torch is None:
        return {"available": False, "backend": None, "devices": []}

    try:
        if torch.cuda.is_available():
            devices = []
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                devices.append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_total_mb": round(total / 1024**2, 1),
                        "memory_used_mb": round((total - free) / 1024**2, 1),
                    }
                )
            return {"available": True, "backend": "cuda", "devices": devices}

        # Apple Silicon
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return {
                "available": True,
                "backend": "mps",
                "devices": [{"index": 0, "name": "Apple Metal (MPS)"}],
            }
    except Exception as exc:
        logger.debug("GPU probe failed: %s", exc)

    return {"available": False, "backend": "cpu", "devices": []}
