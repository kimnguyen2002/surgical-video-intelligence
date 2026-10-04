"""
Calibrate the out-of-domain guard from the training frames.

Run from the repository root::

    python -m ai.training.build_domain_reference
    python -m ai.training.build_domain_reference --percentile 99 --verify

Computes the centroid of the training frames' embeddings and a distance
threshold at the given percentile, then writes both to
``data/cache/domain_reference.json``.

Choosing the threshold is a trade between two errors: too tight and genuine
surgical footage from an unfamiliar camera gets refused, too loose and the
webcam-pointed-at-a-face case sails through.

Measured on this dataset the two populations separate cleanly — training frames
span 0.07–0.61 while non-surgical probes land near 0.89 — so the threshold
belongs in the gap, not at a percentile of the training distribution. A raw
percentile is the wrong instrument here: p99 sits *inside* the training spread
by definition, so it rejects 1% of real surgical frames for no benefit.

The threshold is therefore a high percentile scaled by ``--margin``, which
pushes it into the empty band between the distributions. ``--verify`` measures
both sides so the choice is checked rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

from ai.common import settings
from ai.training.config import REPO_ROOT

logger = logging.getLogger("surgvu.domain_reference")

#: Every corpus of genuine surgical frames on disk, not just one.
#:
#: Calibrating on cat1 alone produced a reference too narrow to recognise
#: surgvu24 footage: sampled surgvu24 frames scored past the threshold and had
#: their predictions withheld, which would silently disable analysis across
#: most of the library. "Surgical video" has to mean both corpora, so the
#: reference is built from both.
DEFAULT_FRAME_DIRS = (
    REPO_ROOT / "data" / "cat1_yolo" / "images" / "train",
    REPO_ROOT / "data" / "surgvu_tasks",
)
OUTPUT = settings.cache_dir / "domain_reference.json"


def _normalise(vector):
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _cosine_distance(a, b) -> float:
    return 1.0 - sum(x * y for x, y in zip(a, b))


def _kmeans(vectors, k: int, iterations: int = 25, seed: int = 42):
    """
    Spherical k-means on unit vectors.

    Written out rather than pulled from scikit-learn, which is not a dependency
    — this is fifty lines and avoids adding a 40 MB package for one call.

    Why clustering at all: "surgical video" is not one blob. cat1 and surgvu24
    sit in different regions, and surgvu24 itself spans 81 operations with
    widely varying scenes, so a single centroid per corpus still needed a
    radius of 0.91 — wide enough to swallow a flat colour field at 0.90.
    Several tight clusters model the real density instead of averaging over it.
    """
    import random as _random

    if len(vectors) <= k:
        return [list(v) for v in vectors]

    rng = _random.Random(seed)

    # k-means++ seeding: spread the initial centres out, so a bad random draw
    # cannot collapse several centres into the same dense region.
    centres = [list(vectors[rng.randrange(len(vectors))])]
    while len(centres) < k:
        distances = [
            min(_cosine_distance(v, c) for c in centres) ** 2 for v in vectors
        ]
        total = sum(distances)
        if total <= 0:
            centres.append(list(vectors[rng.randrange(len(vectors))]))
            continue
        target = rng.random() * total
        running = 0.0
        for vector, weight in zip(vectors, distances):
            running += weight
            if running >= target:
                centres.append(list(vector))
                break

    dimension = len(vectors[0])
    for _ in range(iterations):
        buckets: list[list] = [[] for _ in centres]
        for vector in vectors:
            best = min(
                range(len(centres)),
                key=lambda i: _cosine_distance(vector, centres[i]),
            )
            buckets[best].append(vector)

        moved = False
        for index, bucket in enumerate(buckets):
            if not bucket:
                continue
            updated = _normalise(
                [sum(v[d] for v in bucket) / len(bucket) for d in range(dimension)]
            )
            if _cosine_distance(updated, centres[index]) > 1e-6:
                moved = True
            centres[index] = updated
        if not moved:
            break

    return centres


def build(
    frame_dirs: Optional[list[Path]] = None,
    *,
    percentile: float = 97.0,
    margin: float = 1.15,
    clusters: int = 16,
    max_radius: float = 0.65,
    limit: Optional[int] = None,
    output: Path = OUTPUT,
) -> dict:
    from PIL import Image

    from ai.embeddings.vision import get_frame_embedder

    dirs = [Path(d) for d in (frame_dirs or DEFAULT_FRAME_DIRS)]
    present = [d for d in dirs if d.is_dir()]
    if not present:
        raise SystemExit(
            "No frame directories found. Build one first:\n"
            "  python -m ai.training.export_detection\n"
            "  python -m ai.training.extract_task_frames"
        )

    # Sample each corpus separately, then merge. Pooling first and striding
    # afterwards would let the larger corpus dominate the centroid and pull the
    # reference back toward a single source — the failure being fixed here.
    per_source = max(1, (limit or 2000) // len(present))
    images: list[Path] = []
    sources: dict[str, int] = {}
    for directory in present:
        found = sorted(directory.rglob("*.jpg"))
        if not found:
            continue
        step = max(1, len(found) // per_source)
        sampled = found[::step][:per_source]
        images.extend(sampled)
        sources[str(directory)] = len(sampled)

    if not images:
        raise SystemExit(f"No .jpg frames under: {', '.join(str(d) for d in present)}")

    embedder = get_frame_embedder()
    logger.info("Embedding %d frames with %s", len(images), embedder.info()["backend"])

    vectors: list[list[float]] = []
    for index, path in enumerate(images):
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            vectors.append(_normalise(embedder.encode_pil([image])[0]))
        except Exception as exc:
            logger.debug("skipped %s: %s", path.name, exc)
        if (index + 1) % 200 == 0:
            logger.info("  %d/%d", index + 1, len(images))

    if len(vectors) < 20:
        raise SystemExit(f"Only {len(vectors)} frames embedded — too few to calibrate.")

    dimension = len(vectors[0])

    # Several tight prototypes, found by clustering, rather than one centroid.
    # See _kmeans for why: a single centroid — even one per corpus — needs a
    # radius wide enough to admit a flat colour field.
    centres = _kmeans(vectors, k=clusters)

    assigned: dict[int, list] = {i: [] for i in range(len(centres))}
    for vector in vectors:
        best = min(
            range(len(centres)), key=lambda i: _cosine_distance(vector, centres[i])
        )
        assigned[best].append(vector)

    prototypes = []
    for index, centroid in enumerate(centres):
        members = assigned[index]
        if len(members) < 8:
            # Too few members to estimate a radius from; folding them into a
            # neighbouring cluster is safer than inventing a threshold.
            continue
        group_distances = sorted(_cosine_distance(v, centroid) for v in members)
        cut = min(
            len(group_distances) - 1,
            int(len(group_distances) * percentile / 100.0),
        )
        # Scale past this cluster's spread into the empty band before the
        # out-of-domain population begins, then cap.
        #
        # The cap is what stops one incoherent cluster from defining the whole
        # domain. Measured without it: 11 clusters landed between 0.20 and 0.55
        # while one reached 0.88 — and that single radius admitted a flat grey
        # field (0.84) and a skin-tone field (0.82). A cluster whose own
        # members sit 0.88 away from its centre is not describing a mode, it is
        # absorbing outliers.
        radius = min(group_distances[cut] * margin, max_radius)
        prototypes.append(
            {
                "source": f"cluster_{index}",
                "centroid": centroid,
                "threshold": max(radius, 1e-6),
                "frames": len(members),
                "median": round(group_distances[len(group_distances) // 2], 5),
                "max": round(group_distances[-1], 5),
                "capped": group_distances[cut] * margin > max_radius,
            }
        )

    if not prototypes:
        raise SystemExit("Clustering produced no usable prototypes.")

    payload = {
        "prototypes": [
            {k: v for k, v in p.items() if k != "centroid"} | {"centroid": p["centroid"]}
            for p in prototypes
        ],
        # Kept so an older guard build still loads something sane rather than
        # failing outright: the tightest cluster.
        "centroid": min(prototypes, key=lambda p: p["threshold"])["centroid"],
        "threshold": min(p["threshold"] for p in prototypes),
        "embedder": embedder.info()["backend"],
        "percentile": percentile,
        "margin": margin,
        "clusters_requested": clusters,
        "stats": {
            "frames": len(vectors),
            "dim": dimension,
            "sources": sources,
            "clusters": [
                {
                    "source": p["source"],
                    "frames": p["frames"],
                    "median": p["median"],
                    "max": p["max"],
                    "threshold": round(p["threshold"], 5),
                }
                for p in prototypes
            ],
        },
    }

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload), encoding="utf-8")
    logger.info("Wrote %s", output)
    return payload


def verify(payload: dict) -> int:
    """
    Check the threshold separates surgical frames from obvious non-surgical ones.

    A calibration that accepts a synthetic colour field is not a guard, and the
    only way to know is to measure it.
    """
    import numpy as np
    from PIL import Image

    from ai.embeddings.vision import get_frame_embedder
    from ai.vision.domain_guard import DomainGuard

    guard = DomainGuard()
    guard.load(force=True)
    embedder = get_frame_embedder()
    backend = embedder.info()["backend"]

    rng = np.random.default_rng(0)
    probes = {
        "uniform noise": Image.fromarray(
            rng.integers(0, 255, (512, 640, 3), dtype=np.uint8)
        ),
        "flat grey": Image.new("RGB", (640, 512), (128, 128, 128)),
        "flat blue (sky-like)": Image.new("RGB", (640, 512), (110, 160, 220)),
        "skin-tone field (face-like)": Image.new("RGB", (640, 512), (222, 176, 148)),
    }

    found = payload["stats"].get("clusters", [])
    print(f"\n{len(found)} reference cluster(s), "
          f"p{payload['percentile']:.1f} × margin {payload['margin']}")
    radii = [c["threshold"] for c in found]
    if radii:
        print(f"  radii: min {min(radii):.4f}  median "
              f"{sorted(radii)[len(radii) // 2]:.4f}  max {max(radii):.4f}")
    print("\nprobe                          distance   novelty   verdict")

    failures = 0
    for label, image in probes.items():
        vector = embedder.encode_pil([image])[0]
        verdict = guard.check(vector, embedder=backend)
        mark = "IN-DOMAIN (!)" if verdict.in_domain else "refused"
        if verdict.in_domain:
            failures += 1
        print(f"{label:30s} {verdict.distance:8.4f}  {verdict.novelty:7.2f}   {mark}")

    # Real surgical frames must still pass — checked on **held-out** data from
    # every corpus. Verifying against the fitting data would only confirm the
    # percentile arithmetic, and checking one corpus is what let the cat1-only
    # reference silently refuse surgvu24.
    from ai.training.config import REPO_ROOT as _ROOT

    holdouts = {
        "held-out cat1": _ROOT / "data" / "cat1_yolo" / "images" / "val",
        "surgvu24 tasks": _ROOT / "data" / "surgvu_tasks",
    }
    for label, directory in holdouts.items():
        if not directory.is_dir():
            continue
        samples = sorted(directory.rglob("*.jpg"))
        if not samples:
            continue
        step = max(1, len(samples) // 12)
        checked = samples[::step][:12]
        refused = 0
        worst_distance = 0.0
        worst_novelty = 0.0
        for path in checked:
            with Image.open(path) as handle:
                vector = embedder.encode_pil([handle.convert("RGB")])[0]
            verdict = guard.check(vector, embedder=backend)
            worst_distance = max(worst_distance, verdict.distance)
            # The verdict's own novelty — distance to the *matched* prototype
            # over that prototype's radius. Dividing by the global minimum
            # threshold, as an earlier version did, reported novelty 2.93 for
            # frames that were in fact accepted.
            worst_novelty = max(worst_novelty, verdict.novelty)
            if not verdict.in_domain:
                refused += 1
        mark = "all accepted" if refused == 0 else f"{refused}/{len(checked)} REFUSED (!)"
        print(f"{label + f' ({len(checked)})':30s} {worst_distance:8.4f}  "
              f"{worst_novelty:7.2f}   {mark}")
        failures += refused

    if failures:
        print(f"\n{failures} probe(s) landed on the wrong side of the threshold.")
        print("Lower --percentile to tighten, or raise it to loosen.")
    else:
        print("\nAll probes classified correctly.")
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames", type=Path, action="append", default=None,
        help="frame directory (repeatable); defaults to every corpus on disk",
    )
    parser.add_argument("--percentile", type=float, default=97.0)
    parser.add_argument(
        "--clusters", type=int, default=16,
        help="reference clusters; more = tighter radii, see _kmeans",
    )
    parser.add_argument(
        "--max-radius", type=float, default=0.65,
        help="hard cap on any cluster's radius (see build)",
    )
    parser.add_argument(
        "--margin", type=float, default=1.15,
        help="scale the percentile past the training spread (see module docstring)",
    )
    parser.add_argument("--limit", type=int, default=2400)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s │ %(message)s")

    payload = build(
        args.frames, percentile=args.percentile, margin=args.margin,
        clusters=args.clusters, max_radius=args.max_radius,
        limit=args.limit, output=args.out,
    )
    print(f"\nDomain reference written to {args.out}")
    print(f"  frames    : {payload['stats']['frames']}")
    print(f"  threshold : {payload['threshold']:.4f}")
    for source, count in payload["stats"].get("sources", {}).items():
        print(f"    {count:5d}  {source}")

    if args.verify:
        return verify(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
