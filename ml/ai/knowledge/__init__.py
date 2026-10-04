"""Specialty ontologies and the surgical knowledge graph built from them."""

from .graph import KnowledgeGraph, knowledge_graph
from .ontologies import ONTOLOGIES, SPECIALTIES, get_ontology
from .risks import DANGER_ZONES, coverage, phases_with_risks, risks_for

__all__ = [
    "KnowledgeGraph",
    "knowledge_graph",
    "ONTOLOGIES",
    "SPECIALTIES",
    "get_ontology",
    "DANGER_ZONES",
    "risks_for",
    "phases_with_risks",
    "coverage",
]
