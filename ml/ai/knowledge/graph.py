"""
The surgical knowledge graph.

Ontologies are compiled into a typed multigraph: phases, anatomy, instruments,
events, and concepts become nodes, and the relationships between them become
edges (``precedes``, ``involves``, ``uses``, ``may_produce``, ``teaches``).

The graph is kept deliberately separate from the vision models. It supplies
terminology, workflow expectations, and relational context for retrieval and
explanation, and it can be extended or replaced for a new specialty without
retraining or even touching a single model.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .ontologies import ONTOLOGIES, SPECIALTIES, get_ontology


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    kind: str  # specialty | phase | anatomy | instrument | event | concept
    specialty: str
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "specialty": self.specialty,
            "description": self.description,
        }


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    relation: str

    def to_dict(self) -> dict:
        return {"source": self.source, "target": self.target, "relation": self.relation}


@dataclass
class Subgraph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }


def _slug(*parts: str) -> str:
    raw = "/".join(parts).lower()
    return re.sub(r"[^a-z0-9]+", "-", raw).strip("-")


class KnowledgeGraph:
    """A queryable graph built from every specialty ontology."""

    def __init__(self):
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._out: dict[str, list[Edge]] = defaultdict(list)
        self._in: dict[str, list[Edge]] = defaultdict(list)
        self._build()

    # -- construction ------------------------------------------------------
    def _add_node(self, node: Node) -> str:
        self._nodes.setdefault(node.id, node)
        return node.id

    def _add_edge(self, source: str, target: str, relation: str) -> None:
        edge = Edge(source, target, relation)
        self._edges.append(edge)
        self._out[source].append(edge)
        self._in[target].append(edge)

    def _build(self) -> None:
        for specialty, ontology in ONTOLOGIES.items():
            spec_id = self._add_node(
                Node(
                    id=_slug("specialty", specialty),
                    label=specialty,
                    kind="specialty",
                    specialty=specialty,
                    description=ontology.get("description", ""),
                )
            )

            phase_ids: list[str] = []
            for phase in ontology.get("phases", []):
                phase_id = self._add_node(
                    Node(
                        id=_slug(specialty, "phase", phase["name"]),
                        label=phase["name"],
                        kind="phase",
                        specialty=specialty,
                        description=phase.get("description", ""),
                    )
                )
                phase_ids.append(phase_id)
                self._add_edge(spec_id, phase_id, "has_phase")

                for landmark in phase.get("landmarks", []):
                    landmark_id = self._add_node(
                        Node(
                            id=_slug(specialty, "anatomy", landmark),
                            label=landmark,
                            kind="anatomy",
                            specialty=specialty,
                        )
                    )
                    self._add_edge(phase_id, landmark_id, "involves")

            # Workflow order — the expected progression through the operation.
            for earlier, later in zip(phase_ids, phase_ids[1:]):
                self._add_edge(earlier, later, "precedes")

            for anatomy in ontology.get("anatomy", []):
                anatomy_id = self._add_node(
                    Node(
                        id=_slug(specialty, "anatomy", anatomy),
                        label=anatomy,
                        kind="anatomy",
                        specialty=specialty,
                    )
                )
                self._add_edge(spec_id, anatomy_id, "has_anatomy")

            for instrument in ontology.get("instruments", []):
                instrument_id = self._add_node(
                    Node(
                        id=_slug(specialty, "instrument", instrument),
                        label=instrument,
                        kind="instrument",
                        specialty=specialty,
                    )
                )
                self._add_edge(spec_id, instrument_id, "uses")

            for event in ontology.get("events", []):
                event_id = self._add_node(
                    Node(
                        id=_slug(specialty, "event", event),
                        label=event,
                        kind="event",
                        specialty=specialty,
                    )
                )
                self._add_edge(spec_id, event_id, "may_produce")

            for i, concept in enumerate(ontology.get("concepts", [])):
                concept_id = self._add_node(
                    Node(
                        id=_slug(specialty, "concept", str(i)),
                        label=concept,
                        kind="concept",
                        specialty=specialty,
                        description=concept,
                    )
                )
                self._add_edge(spec_id, concept_id, "teaches")

    # -- queries -----------------------------------------------------------
    def specialties(self) -> list[str]:
        return SPECIALTIES

    def node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def neighbors(self, node_id: str) -> list[tuple[str, Node]]:
        """Adjacent nodes in both directions, with the relation that links them."""
        out = [
            (e.relation, self._nodes[e.target])
            for e in self._out.get(node_id, [])
            if e.target in self._nodes
        ]
        incoming = [
            (f"inverse:{e.relation}", self._nodes[e.source])
            for e in self._in.get(node_id, [])
            if e.source in self._nodes
        ]
        return out + incoming

    def phases(self, specialty: str) -> list[dict]:
        return get_ontology(specialty).get("phases", [])

    def phase_names(self, specialty: str) -> list[str]:
        return [p["name"] for p in self.phases(specialty)]

    def next_phase(self, specialty: str, current: str) -> Optional[str]:
        """The phase expected to follow the current one."""
        names = self.phase_names(specialty)
        for i, name in enumerate(names[:-1]):
            if name.lower() == (current or "").lower():
                return names[i + 1]
        return None

    def progress(self, specialty: str, current: str) -> Optional[float]:
        """
        Rough procedural progress, 0–1, from position in the phase order.

        This is an ordinal estimate: phases are not equal in duration, so it
        reflects "how far through the sequence" rather than elapsed time.
        """
        names = self.phase_names(specialty)
        if not names:
            return None
        for i, name in enumerate(names):
            if name.lower() == (current or "").lower():
                return round((i + 1) / len(names), 3)
        return None

    def subgraph(self, specialty: str, kinds: Optional[Iterable[str]] = None) -> Subgraph:
        """The graph for one specialty, optionally restricted to node kinds."""
        wanted = set(kinds) if kinds else None
        nodes = [
            n
            for n in self._nodes.values()
            if n.specialty == specialty and (wanted is None or n.kind in wanted)
        ]
        ids = {n.id for n in nodes}
        edges = [e for e in self._edges if e.source in ids and e.target in ids]
        return Subgraph(nodes=nodes, edges=edges)

    def search(
        self, query: str, specialty: Optional[str] = None, limit: int = 12
    ) -> list[Node]:
        """Lexical search over node labels and descriptions."""
        terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}
        if not terms:
            return []
        scored: list[tuple[float, Node]] = []
        for node in self._nodes.values():
            if specialty and node.specialty != specialty:
                continue
            haystack = f"{node.label} {node.description}".lower()
            hits = sum(1 for t in terms if t in haystack)
            if not hits:
                continue
            # Exact label matches should outrank incidental description hits.
            bonus = 2.0 if node.label.lower() in query.lower() else 0.0
            scored.append((hits + bonus, node))
        scored.sort(key=lambda pair: -pair[0])
        return [node for _, node in scored[:limit]]

    def context_for(
        self, specialty: str, phase: Optional[str] = None, tools: Optional[list[str]] = None
    ) -> str:
        """
        Compact, factual context about the current situation, injected into the
        assistant's prompt so its vocabulary matches the specialty.
        """
        ontology = get_ontology(specialty)
        lines = [f"Specialty: {specialty} — {ontology.get('description', '')}"]

        names = [p["name"] for p in ontology.get("phases", [])]
        if names:
            lines.append("Typical workflow: " + " → ".join(names))

        if phase:
            match = next(
                (p for p in ontology.get("phases", []) if p["name"].lower() == phase.lower()),
                None,
            )
            if match:
                lines.append(f"Current phase '{match['name']}': {match['description']}")
                if match.get("landmarks"):
                    lines.append("Landmarks: " + ", ".join(match["landmarks"]))
            following = self.next_phase(specialty, phase)
            if following:
                lines.append(f"Normally followed by: {following}")
            progress = self.progress(specialty, phase)
            if progress is not None:
                lines.append(f"Ordinal progress through the phase sequence: {progress:.0%}")

        if tools:
            lines.append("Instruments currently detected: " + ", ".join(tools))

        if ontology.get("concepts"):
            lines.append("Teaching points: " + " ".join(ontology["concepts"]))

        return "\n".join(lines)

    def stats(self) -> dict:
        by_kind: dict[str, int] = defaultdict(int)
        for node in self._nodes.values():
            by_kind[node.kind] += 1
        return {
            "specialties": len(SPECIALTIES),
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "by_kind": dict(by_kind),
        }


#: Process-wide graph — construction is cheap and purely in-memory.
knowledge_graph = KnowledgeGraph()
