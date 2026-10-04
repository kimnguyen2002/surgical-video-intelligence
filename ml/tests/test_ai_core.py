"""Vector store, embeddings, chunking, temporal segmentation, and RAG."""

from __future__ import annotations

import math

import pytest

from ai.embeddings import get_text_embedder
from ai.embeddings.text import cosine
from ai.rag.chunker import chunk_text, split_sentences
from ai.rag.documents import extract_text
from ai.temporal_memory import EventSegmenter, FrameObservation
from ai.vectorstore import VectorStore


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
def test_vector_store_roundtrip(data_dir):
    store = VectorStore("test_roundtrip", dim=4, directory=data_dir / "vec")
    store.clear()

    store.add("a", [1, 0, 0, 0], {"kind": "x"}, text="alpha")
    store.add("b", [0, 1, 0, 0], {"kind": "y"}, text="beta")
    store.add("c", [0.9, 0.1, 0, 0], {"kind": "x"}, text="gamma")

    results = store.search([1, 0, 0, 0], top_k=2)
    assert results[0].id == "a"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[1].id == "c"


def test_vector_store_normalises_on_write():
    """Scores must be comparable regardless of input magnitude."""
    store = VectorStore("test_norm", dim=3)
    store.clear()
    store.add("small", [1, 0, 0], {})
    store.add("large", [1000, 0, 0], {})

    results = store.search([5, 0, 0], top_k=2)
    assert results[0].score == pytest.approx(results[1].score, abs=1e-5)


def test_vector_store_metadata_filter():
    store = VectorStore("test_filter", dim=3)
    store.clear()
    store.add("a", [1, 0, 0], {"group": "keep"})
    store.add("b", [1, 0, 0], {"group": "drop"})

    results = store.search([1, 0, 0], top_k=5, where={"group": "keep"})
    assert [r.id for r in results] == ["a"]


def test_vector_store_persists_across_instances(data_dir):
    directory = data_dir / "persist"
    first = VectorStore("test_persist", dim=3, directory=directory)
    first.clear()
    first.add("x", [1, 0, 0], {"note": "hello"}, text="content")

    second = VectorStore("test_persist", dim=3, directory=directory)
    assert len(second) == 1
    assert second.get("x").metadata["note"] == "hello"


def test_vector_store_delete_compacts():
    store = VectorStore("test_delete", dim=3)
    store.clear()
    store.add_many([(f"id{i}", [i, 1, 0], {}, "") for i in range(5)])
    assert store.delete(["id1", "id3"]) == 2
    assert len(store) == 3
    assert store.get("id1") is None
    # Remaining ids must still be searchable after compaction.
    assert store.search([4, 1, 0], top_k=1)[0].id == "id4"


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def test_text_embeddings_are_unit_norm_and_deterministic():
    embedder = get_text_embedder()
    a = embedder.encode_one("cystic artery dissection")
    b = embedder.encode_one("cystic artery dissection")

    assert a == b
    assert math.sqrt(sum(x * x for x in a)) == pytest.approx(1.0, abs=1e-5)


def test_similar_text_scores_above_unrelated_text():
    embedder = get_text_embedder()
    query = embedder.encode_one("bleeding from the cystic artery")
    related = embedder.encode_one("the cystic artery was bleeding heavily")
    unrelated = embedder.encode_one("kubernetes ingress controller configuration")

    assert cosine(query, related) > cosine(query, unrelated)


def test_empty_text_does_not_crash():
    assert get_text_embedder().encode([]) == []
    assert len(get_text_embedder().encode_one("")) > 0


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def test_sentence_split_keeps_abbreviations_intact():
    sentences = split_sentences("See Fig. 3 for detail. The duct was clipped.")
    assert len(sentences) == 2
    assert "Fig. 3" in sentences[0]


def test_chunking_covers_all_content():
    text = " ".join(f"Sentence number {i} about surgical anatomy." for i in range(60))
    chunks = chunk_text(text, chunk_size=200, overlap=40)

    assert len(chunks) > 1
    assert all(c.text.strip() for c in chunks)
    # Overlap must not stall the cursor and loop forever.
    assert len(chunks) < 100
    assert "Sentence number 0" in chunks[0].text
    assert "Sentence number 59" in chunks[-1].text


def test_oversized_sentence_is_hard_split():
    chunks = chunk_text("word " * 500, chunk_size=100, overlap=20)
    assert all(len(c.text) <= 120 for c in chunks)


def test_chunk_preserves_page_number():
    chunks = chunk_text("A sentence about anatomy here.", page=7)
    assert chunks[0].page == 7


# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------
def test_markdown_parses(sample_document):
    parsed = extract_text("notes.md", sample_document)
    assert parsed.error is None
    assert "cystic artery" in parsed.text


def test_html_strips_script_bodies():
    html = b"<html><script>var secret = 'leak';</script><p>Visible prose.</p></html>"
    parsed = extract_text("page.html", html)
    assert "Visible prose." in parsed.text
    assert "secret" not in parsed.text


def test_unsupported_format_reports_clearly():
    parsed = extract_text("model.bin", b"\x00\x01\x02")
    assert parsed.error and "Unsupported" in parsed.error


# ---------------------------------------------------------------------------
# Temporal segmentation
# ---------------------------------------------------------------------------
def _observation(t: float, phase: str, vector: list[float]) -> FrameObservation:
    return FrameObservation(timestamp=t, embedding=vector, phase=phase, phase_confidence=0.9)


def test_segmenter_groups_contiguous_phases():
    segmenter = EventSegmenter(drift_threshold=0.9, confirm_frames=1)
    for t in range(0, 10, 2):
        segmenter.observe(_observation(t, "Dissection", [1, 0]))
    for t in range(10, 20, 2):
        segmenter.observe(_observation(t, "Hemostasis", [0, 1]))

    events = segmenter.finish()
    assert [e["phase"] for e in events] == ["Dissection", "Hemostasis"]


def test_segmenter_suppresses_single_frame_flicker():
    """
    One noisy frame between stable predictions must not split the event —
    this is what keeps a real timeline navigable.
    """
    segmenter = EventSegmenter(drift_threshold=0.9, confirm_frames=2)
    for t in range(0, 10, 2):
        segmenter.observe(_observation(t, "Dissection", [1, 0]))
    segmenter.observe(_observation(10, "Clipping", [1, 0]))  # single-frame blip
    for t in range(12, 22, 2):
        segmenter.observe(_observation(t, "Dissection", [1, 0]))

    events = segmenter.finish()
    assert len(events) == 1
    assert events[0]["phase"] == "Dissection"


def test_segmenter_splits_on_visual_drift():
    segmenter = EventSegmenter(drift_threshold=0.3, confirm_frames=1)
    for t in range(0, 6, 2):
        segmenter.observe(_observation(t, "Dissection", [1, 0]))
    for t in range(6, 12, 2):
        segmenter.observe(_observation(t, "Dissection", [0, 1]))  # scene changed

    assert len(segmenter.finish()) == 2


def test_segmenter_handles_no_observations():
    assert EventSegmenter().finish() == []
