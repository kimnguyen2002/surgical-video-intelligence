"""
Provider selection.

Order of preference in ``auto`` mode:

1. **Ollama** — fully local, no data leaves the machine. This is the default
   the platform is designed around.
2. **Gemini** — only if ``GEMINI_API_KEY`` is set (the AI Studio path).
3. **Knowledge** — always available, extractive, clearly labelled.

Selection is re-evaluated periodically so starting Ollama mid-session upgrades
the assistant without a restart.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from ai.common import settings

from .base import LLMProvider
from .gemini import GeminiProvider
from .knowledge import KnowledgeProvider
from .ollama import OllamaProvider

logger = logging.getLogger("charlie.llm.registry")

_LOCK = threading.Lock()
_PROVIDERS: dict[str, LLMProvider] = {}
_ACTIVE: Optional[LLMProvider] = None
_RESOLVED_AT = 0.0
_TTL = 30.0


def _all() -> dict[str, LLMProvider]:
    global _PROVIDERS
    if not _PROVIDERS:
        _PROVIDERS = {
            "ollama": OllamaProvider(),
            "gemini": GeminiProvider(),
            "knowledge": KnowledgeProvider(),
        }
    return _PROVIDERS


def get_provider(preference: Optional[str] = None) -> LLMProvider:
    """Return the provider to answer with."""
    global _ACTIVE, _RESOLVED_AT

    requested = (preference or settings.llm_provider or "auto").lower()
    providers = _all()

    if requested != "auto":
        provider = providers.get(requested)
        if provider is not None and provider.is_available():
            return provider
        if provider is not None:
            logger.warning(
                "Requested LLM provider '%s' is unavailable; falling back.", requested
            )

    with _LOCK:
        if _ACTIVE is not None and time.time() - _RESOLVED_AT < _TTL:
            return _ACTIVE
        for name in ("ollama", "gemini", "knowledge"):
            provider = providers[name]
            if provider.is_available():
                if _ACTIVE is None or _ACTIVE.name != name:
                    logger.info("LLM provider: %s (%s)", name, provider.model)
                _ACTIVE = provider
                _RESOLVED_AT = time.time()
                return provider
        _ACTIVE = providers["knowledge"]
        _RESOLVED_AT = time.time()
        return _ACTIVE


def available_providers() -> list[dict]:
    """Status of every provider — rendered in the developer console."""
    active = get_provider()
    out = []
    for name, provider in _all().items():
        info = provider.info()
        info["active"] = provider is active
        out.append(info)
    return out


def reset_provider() -> None:
    """Force re-detection (used after config changes and in tests)."""
    global _ACTIVE, _RESOLVED_AT, _PROVIDERS
    with _LOCK:
        _ACTIVE = None
        _RESOLVED_AT = 0.0
        _PROVIDERS = {}
