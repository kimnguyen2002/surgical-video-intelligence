"""
Central runtime configuration for the AI subsystem.

Everything is overridable by environment variable so the same image runs
unchanged on a laptop, in Docker Compose, and on Kubernetes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repository root — this file lives at <root>/ai/common/settings.py
REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # --- Storage ---------------------------------------------------------
    data_dir: Path = field(
        default_factory=lambda: Path(_env("CHARLIE_DATA_DIR", str(REPO_ROOT / "data")))
    )

    # --- Embeddings ------------------------------------------------------
    text_embedding_model: str = field(
        default_factory=lambda: _env("CHARLIE_TEXT_EMBED_MODEL", "all-MiniLM-L6-v2")
    )
    #: Dimensionality of the offline hashed fallback embedder.
    hashed_embedding_dim: int = field(
        default_factory=lambda: _env_int("CHARLIE_HASHED_EMBED_DIM", 384)
    )
    vision_embedding_model: str = field(
        default_factory=lambda: _env("CHARLIE_VISION_EMBED_MODEL", "convnext_tiny")
    )

    # --- Vector store ----------------------------------------------------
    vector_backend: str = field(
        default_factory=lambda: _env("CHARLIE_VECTOR_BACKEND", "auto")
    )  # auto | faiss | numpy | python

    # --- RAG -------------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _env_int("CHARLIE_CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHARLIE_CHUNK_OVERLAP", 150))
    rag_top_k: int = field(default_factory=lambda: _env_int("CHARLIE_RAG_TOP_K", 5))
    rag_min_score: float = field(
        default_factory=lambda: _env_float("CHARLIE_RAG_MIN_SCORE", 0.12)
    )

    # --- Temporal memory -------------------------------------------------
    #: Seconds between sampled frames when indexing a procedure.
    memory_sample_interval: float = field(
        default_factory=lambda: _env_float("CHARLIE_MEMORY_SAMPLE_INTERVAL", 2.0)
    )
    #: Cosine distance above which a new frame opens a new memory event.
    memory_event_threshold: float = field(
        default_factory=lambda: _env_float("CHARLIE_MEMORY_EVENT_THRESHOLD", 0.25)
    )

    # --- LLM -------------------------------------------------------------
    llm_provider: str = field(
        default_factory=lambda: _env("CHARLIE_LLM_PROVIDER", "auto")
    )  # auto | ollama | gemini | knowledge
    ollama_host: str = field(
        default_factory=lambda: _env("OLLAMA_HOST", "http://127.0.0.1:11434")
    )
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:7b"))
    gemini_api_key: str = field(
        default_factory=lambda: _env("GEMINI_API_KEY", "") or _env("GOOGLE_API_KEY", "")
    )
    gemini_model: str = field(
        default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.0-flash")
    )
    llm_temperature: float = field(
        default_factory=lambda: _env_float("CHARLIE_LLM_TEMPERATURE", 0.4)
    )
    #: 400, not 900.
    #:
    #: A local model on CPU generates roughly 15 tokens/second, so a 900-token
    #: ceiling permits a 60-second answer — measured at 29 s for a 450-token
    #: reply on qwen2.5:1.5b. Nobody reads 900 tokens of chat, and the length
    #: was buying degeneration rather than depth: the long answers were where
    #: the small model started repeating itself.
    #:
    #: Factual questions about the video do not pass through here at all —
    #: they are answered from annotations in tens of milliseconds.
    llm_max_tokens: int = field(default_factory=lambda: _env_int("CHARLIE_LLM_MAX_TOKENS", 400))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("CHARLIE_LLM_TIMEOUT", 60.0))

    # --- Voice -----------------------------------------------------------
    #: Multilingual by default.
    #:
    #: The English-only `base.en` was marginally more accurate on English and
    #: completely unable to transcribe anything else — it silently returns
    #: English-looking nonsense for Vietnamese or Chinese speech rather than
    #: failing, which is the worst way for a language barrier to appear. `base`
    #: is the same size and covers 99 languages.
    stt_model: str = field(default_factory=lambda: _env("CHARLIE_STT_MODEL", "base"))
    #: Empty means auto-detect per utterance. Pin it to a code (`en`, `vi`,
    #: `zh`) when the language is known — detection costs a little accuracy on
    #: short commands, which is most of what gets spoken here.
    stt_language: str = field(default_factory=lambda: _env("CHARLIE_STT_LANGUAGE", ""))
    tts_voice_model: str = field(default_factory=lambda: _env("CHARLIE_TTS_VOICE", ""))
    tts_language: str = field(default_factory=lambda: _env("CHARLIE_TTS_LANGUAGE", "en-US"))

    # --- Inference -------------------------------------------------------
    tool_checkpoint: str = field(default_factory=lambda: _env("CHARLIE_TOOL_CHECKPOINT", ""))
    step_checkpoint: str = field(default_factory=lambda: _env("CHARLIE_STEP_CHECKPOINT", ""))
    inference_device: str = field(default_factory=lambda: _env("CHARLIE_DEVICE", "auto"))
    enable_gradcam: bool = field(
        default_factory=lambda: _env_bool("CHARLIE_ENABLE_GRADCAM", True)
    )

    # --- Safety ----------------------------------------------------------
    #: Appended to every generated answer. Non-negotiable: this platform is
    #: for education and research, never for clinical decision-making.
    disclaimer: str = (
        "Educational and research use only — not a medical device, and not a "
        "substitute for the judgement of a licensed clinician."
    )

    # --- Derived paths ---------------------------------------------------
    @property
    def vector_dir(self) -> Path:
        return self._ensure(self.data_dir / "vector_db")

    @property
    def documents_dir(self) -> Path:
        return self._ensure(self.data_dir / "documents")

    @property
    def memory_dir(self) -> Path:
        return self._ensure(self.data_dir / "memory")

    @property
    def annotations_dir(self) -> Path:
        return self._ensure(self.data_dir / "annotations")

    @property
    def experiments_dir(self) -> Path:
        return self._ensure(self.data_dir / "experiments")

    @property
    def media_dir(self) -> Path:
        return self._ensure(self.data_dir / "media")

    @property
    def cache_dir(self) -> Path:
        return self._ensure(self.data_dir / "cache")

    @staticmethod
    def _ensure(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
