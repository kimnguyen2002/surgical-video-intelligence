"""
Shared pytest fixtures.

Every test runs against a temporary data directory. The AI singletons
(`rag_index`, `memory_engine`, …) resolve their paths from `ai.common.settings`
at import time, so the environment variable is set *before* any of them is
imported — otherwise a test run would read and write the developer's real
corpus and procedure memory.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "backend")):
    if path not in sys.path:
        sys.path.insert(0, path)

# Must happen before `ai.common.settings` is first imported.
_TEMP_DIR = tempfile.mkdtemp(prefix="charlie-tests-")
os.environ.setdefault("CHARLIE_DATA_DIR", _TEMP_DIR)
# Keep tests off any locally running Ollama so results are deterministic.
os.environ.setdefault("CHARLIE_LLM_PROVIDER", "knowledge")
os.environ.setdefault("GEMINI_API_KEY", "")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_torch: needs PyTorch installed")
    config.addinivalue_line("markers", "slow: takes more than a second")


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(_TEMP_DIR)


# The `client` fixture is gone along with the FastAPI backend it served.
#
# This project no longer has a server: the application is a static page and the
# model code here exists as the provenance for the checkpoints it ships. The
# tests that exercised HTTP endpoints went with it; what remains tests the
# dataset schema, the label parsing and the model pipeline, which is the part
# that still has to be right.


def client():
    """FastAPI test client with the app's lifespan actually run."""
    from fastapi.testclient import TestClient

    from main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_document() -> bytes:
    return (
        b"The critical view of safety requires clearing the hepatocystic triangle "
        b"of fat and fibrous tissue. Two and only two structures should be seen "
        b"entering the gallbladder: the cystic duct and the cystic artery. "
        b"Bleeding from the cystic artery is controlled with clips rather than "
        b"blind cautery, because blind cautery near the triangle risks thermal "
        b"injury to the common bile duct."
    )
