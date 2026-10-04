"""
AI Surgical Assistant — AI subsystem.

Sub-packages:
    common          Shared settings + optional-dependency capability layer
    embeddings      Text and vision embedding backends
    vectorstore     Pluggable vector index (FAISS / NumPy / pure-Python)
    temporal_memory Per-procedure event memory and semantic timeline
    rag             Document ingestion, chunking, retrieval
    llm             Local/remote language model providers
    voice           Speech-to-text and text-to-speech
    knowledge       Specialty ontologies and knowledge graphs
    evaluation      Experiment tracking and model comparison
    training        SurgVU deep learning pipeline (arXiv:2501.09209v1)

Every sub-package is designed to import successfully with **zero** heavy
dependencies installed, falling back to lighter implementations so the
platform runs identically on a laptop and on a GPU server.
"""

__version__ = "0.2.0"
