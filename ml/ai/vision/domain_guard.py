"""
Out-of-domain guard: is this frame actually surgical video?

The failure this exists to prevent is concrete. Point a webcam at a face and
the instrument classifier reports a needle driver at 0.98, and Grad-CAM
obligingly draws a heatmap over the person's cheek. Nothing in the model
objects, because a multi-label sigmoid head has no way to say "I have never
seen anything like this". It was trained only on endoscopic frames, so every
input is scored as though it were one.

On a platform whose entire premise is that a displayed prediction can be
trusted to mean something, that is worse than being wrong — it is confidently
wrong on input the model has no competence over, and it looks identical to a
correct prediction.

Method
------
Nearest-prototype novelty detection on frame embeddings. Each reference corpus
contributes a centroid and its own radius; a frame is in-domain if it falls
inside *any* of them.

Not a single centroid. cat1 and surgvu24 occupy distinct regions of embedding
space, and one centroid averaged over both lands in the gap between them — so
both corpora sit far from it, and the radius needed to admit them (0.92) also
admits uniform noise (0.91). Measured during calibration, that configuration
accepted a synthetic face. Per-corpus prototypes let each cluster keep a tight
radius, which is what restores the separation.

Cosine distance rather than Mahalanobis: with 768-dimensional embeddings and a
few thousand reference frames the covariance estimate is badly ill-conditioned,
and the inverse it needs amplifies exactly the low-variance directions carrying
least information. The simpler statistic is stable and sufficient for "is this
an operating field at all".

The embedder name is stored with the reference, so changing embedders stands
the guard down rather than silently comparing across incompatible spaces.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ai.common import settings

logger = logging.getLogger("charlie.vision.domain")

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Searched in order.
#:
#: ``checkpoints/`` first because that is where the reference ships in the
#: repository — it is a model artefact, versioned alongside the weights it
#: guards, not a derived cache. ``data/cache/`` is where
#: ``build_domain_reference`` writes when you recalibrate locally, so a fresh
#: calibration overrides the shipped one only if you ask for it there.
REFERENCE_CANDIDATES = (
    REPO_ROOT / "checkpoints" / "domain_reference.json",
    settings.cache_dir / "domain_reference.json",
)


def _default_reference() -> Path:
    for candidate in REFERENCE_CANDIDATES:
        if candidate.exists():
            return candidate
    return REFERENCE_CANDIDATES[0]


REFERENCE_PATH = _default_reference()


@dataclass
class DomainVerdict:
    in_domain: bool
    distance: float
    threshold: float
    #: 0 = indistinguishable from training data, 1 = at the threshold, >1 beyond.
    novelty: float
    reason: Optional[str] = None
    calibrated: bool = True

    def to_dict(self) -> dict:
        return {
            "in_domain": self.in_domain,
            "distance": round(self.distance, 4),
            "threshold": round(self.threshold, 4),
            "novelty": round(self.novelty, 3),
            "reason": self.reason,
            "calibrated": self.calibrated,
        }


def _normalise(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return 1.0 - sum(x * y for x, y in zip(a, b))


class DomainGuard:
    """Loads the reference distribution and scores frames against it."""

    def __init__(self, path: Optional[Path] = None):
        # Resolved per instance, not captured at import: a reference written
        # after the module loaded should be picked up by a later reload().
        self.path = Path(path) if path is not None else _default_reference()
        #: ``(centroid, threshold, source)`` per reference cluster.
        self._prototypes: list[tuple[list[float], float, str]] = []
        self._backend: str = ""
        self._stats: dict = {}
        self._loaded = False

    def load(self, force: bool = False) -> bool:
        if self._loaded and not force:
            return bool(self._prototypes)
        self._loaded = True
        self._prototypes = []
        if force:
            self.path = _default_reference()

        if not self.path.exists():
            logger.info(
                "No domain reference at %s — out-of-domain detection is off. "
                "Build one with: python -m ai.training.build_domain_reference",
                self.path,
            )
            return False

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            entries = payload.get("prototypes")
            if entries:
                for entry in entries:
                    self._prototypes.append(
                        (
                            _normalise(entry["centroid"]),
                            float(entry["threshold"]),
                            str(entry.get("source", "")),
                        )
                    )
            else:
                # Reference written before prototypes existed.
                self._prototypes.append(
                    (_normalise(payload["centroid"]), float(payload["threshold"]), "")
                )
            self._backend = str(payload.get("embedder", ""))
            self._stats = payload.get("stats", {})
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.error("Domain reference at %s is unusable: %s", self.path, exc)
            self._prototypes = []
            return False

        logger.info(
            "Domain guard active (%d cluster(s), %d reference frames, embedder %s)",
            len(self._prototypes), self._stats.get("frames", 0), self._backend or "?",
        )
        return True

    @property
    def available(self) -> bool:
        if not self._loaded:
            self.load()
        return bool(self._prototypes)

    def check(self, embedding: Optional[Sequence[float]], embedder: str = "") -> DomainVerdict:
        """
        Score one frame embedding.

        Returns ``in_domain=True`` when no reference is calibrated. Refusing
        every prediction because the guard was never built would break the
        platform for anyone who has not run the calibration step; the verdict
        carries ``calibrated=False`` so callers can say which case they are in.
        """
        if not self.available or embedding is None:
            return DomainVerdict(
                in_domain=True, distance=0.0, threshold=0.0, novelty=0.0,
                calibrated=False,
                reason=None if embedding is not None else "no embedding",
            )

        # A reference built with a different embedder describes a different
        # space; comparing across them is meaningless, so the guard stands down
        # rather than producing a confident nonsense distance.
        if self._backend and embedder and self._backend != embedder:
            return DomainVerdict(
                in_domain=True, distance=0.0, threshold=0.0, novelty=0.0,
                calibrated=False,
                reason=(
                    f"reference was built with '{self._backend}' but the active "
                    f"embedder is '{embedder}' — rebuild it"
                ),
            )

        vector = _normalise(embedding)

        # Nearest prototype by *relative* distance: clusters have different
        # radii, so comparing raw distances would let a wide cluster claim
        # frames that sit well outside a tight one.
        best_novelty = float("inf")
        best_distance = 0.0
        best_threshold = 0.0
        for centroid, threshold, _source in self._prototypes:
            distance = _cosine_distance(vector, centroid)
            novelty = distance / threshold if threshold > 0 else float("inf")
            if novelty < best_novelty:
                best_novelty, best_distance, best_threshold = novelty, distance, threshold

        distance, threshold, novelty = best_distance, best_threshold, best_novelty
        in_domain = novelty <= 1.0

        return DomainVerdict(
            in_domain=in_domain,
            distance=distance,
            threshold=threshold,
            novelty=novelty,
            reason=(
                None
                if in_domain
                else (
                    "This frame does not resemble the endoscopic surgical video "
                    "the models were trained on, so instrument and phase "
                    "predictions are withheld."
                )
            ),
        )

    def info(self) -> dict:
        """
        A compact summary for the status page.

        Deliberately does not splat ``self._stats``: it carries a ``clusters``
        list of per-cluster records, which collided with the cluster *count*
        here and replaced it with a wall of centroid statistics. Only the
        scalar fields worth displaying are lifted out.
        """
        if not self._loaded:
            self.load()
        thresholds = [t for _c, t, _s in self._prototypes]
        return {
            "available": bool(self._prototypes),
            "cluster_count": len(self._prototypes),
            "radius_min": round(min(thresholds), 4) if thresholds else None,
            "radius_max": round(max(thresholds), 4) if thresholds else None,
            "reference_frames": self._stats.get("frames"),
            "embedder": self._backend,
            "path": str(self.path),
        }


domain_guard = DomainGuard()
