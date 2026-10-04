"""
Calibrate the out-of-domain guard, and expose the embedding it needs.

Why there is a guard at all
---------------------------
Pointing the live camera at a human face made the instrument classifier report
**needle driver at 0.98 confidence**. Nothing was broken. A multi-label
classifier ending in a sigmoid produces a score for every class on every input;
it has no way to represent *"I have never seen anything like this."* It had
only ever seen endoscopic surgical frames, so it scored a face as though it
were one.

That is worse than being wrong. It is confidently wrong on input the model has
no business judging, and on screen it is indistinguishable from a correct
prediction.

Why it is rebuilt here
----------------------
The original guard embedded frames with `timm:convnext_tiny` — a 768-dim,
~110 MB backbone. Shipping that to a browser to answer a yes/no question would
triple the download for a check that runs alongside a model already loaded.

So the guard reuses `tool.onnx`. The tensor feeding its final `Gemm` is the
pooled, layer-normalised feature vector the classifier itself decides on — if
anything in this pipeline knows what a surgical frame looks like, that does.
This script adds that tensor as a second graph **output**, which costs nothing:
the values are already computed on every frame, and the file grows by one
`ValueInfoProto`.

The reference is then the centroid of that embedding over real SurgVU frames,
plus a cosine-distance threshold.

Choosing the threshold
----------------------
A raw percentile of the in-domain distances is the wrong instrument: p99 sits
*inside* the training spread by definition, so it rejects 1% of genuine
surgical frames for no benefit. The two populations separate with a gap between
them, and the threshold belongs in that gap — a high percentile scaled by a
margin. `--verify` measures both sides so the choice is checked rather than
assumed.

    python3 scripts/build_domain_guard.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "public" / "models"
CLIPS = REPO / "public" / "clips"
TOOL_ONNX = MODELS / "tool.onnx"
OUTPUT = MODELS / "domain_reference.json"

#: The tensor feeding the classifier head: pooled and layer-normalised.
EMBEDDING_TENSOR = "/backbone/head/flatten/Flatten_output_0"
EMBEDDING_OUTPUT = "embedding"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: In-domain frames sampled per clip.
FRAMES_PER_CLIP = 60

#: Percentile of in-domain distances the threshold starts from, and the factor
#: that pushes it into the empty band above them.
PERCENTILE = 99.0
MARGIN = 1.15


# ---------------------------------------------------------------------------
# Graph surgery
# ---------------------------------------------------------------------------


def expose_embedding(path: Path) -> bool:
    """Add the pre-head feature tensor as a second graph output, idempotently."""
    model = onnx.load(str(path))
    graph = model.graph

    if any(o.name == EMBEDDING_OUTPUT for o in graph.output):
        print(f"  {path.name}: embedding output already present")
        return False

    producer = next(
        (n for n in graph.node if EMBEDDING_TENSOR in n.output), None
    )
    if producer is None:
        raise SystemExit(
            f"{path.name}: no node produces {EMBEDDING_TENSOR}. The checkpoint was "
            "re-exported with a different graph; re-read the tail of the graph and "
            "update EMBEDDING_TENSOR."
        )

    # Rename the tensor to a stable public name rather than leaving the
    # exporter's internal path as part of the file's contract.
    producer.output[0] = EMBEDDING_OUTPUT
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == EMBEDDING_TENSOR:
                node.input[i] = EMBEDDING_OUTPUT

    graph.output.append(
        onnx.helper.make_tensor_value_info(
            EMBEDDING_OUTPUT, onnx.TensorProto.FLOAT, [1, None]
        )
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    print(f"  {path.name}: added '{EMBEDDING_OUTPUT}' output")
    return True


# ---------------------------------------------------------------------------
# Preprocessing — must match the worker exactly
# ---------------------------------------------------------------------------


def preprocess(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    """
    Scale the shorter side to `size * 1.14`, then take the centre square.

    This mirrors `default_transform(train=False)` in the training pipeline and
    `preprocessClassifier` in the worker. Squashing to a square instead would
    be simpler and wrong: the model never saw horizontally compressed anatomy,
    and the guard would disagree with the classifier it is guarding.
    """
    h, w = frame_bgr.shape[:2]
    scale = (size * 1.14) / min(h, w)
    resized = cv2.resize(
        frame_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR
    )

    rh, rw = resized.shape[:2]
    top, left = max(0, (rh - size) // 2), max(0, (rw - size) // 2)
    crop = resized[top : top + size, left : left + size]
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)


def embed(session: ort.InferenceSession, frames: list[np.ndarray], size: int) -> np.ndarray:
    """L2-normalised embeddings for a list of BGR frames."""
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    index = output_names.index(EMBEDDING_OUTPUT)

    vectors = []
    for frame in frames:
        out = session.run(None, {input_name: preprocess(frame, size)})
        vector = np.asarray(out[index]).reshape(-1).astype(np.float64)
        norm = np.linalg.norm(vector) or 1.0
        vectors.append(vector / norm)
    return np.stack(vectors) if vectors else np.zeros((0, 1))


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------


def in_domain_frames() -> list[np.ndarray]:
    """Real SurgVU frames, sampled evenly across every bundled excerpt."""
    frames: list[np.ndarray] = []
    for clip in json.loads((CLIPS / "clips.json").read_text()):
        capture = cv2.VideoCapture(str(REPO / "public" / clip["file"]))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for index in np.linspace(0, max(0, total - 1), FRAMES_PER_CLIP).astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
        capture.release()
    return frames


def out_of_domain_frames() -> tuple[list[np.ndarray], list[str]]:
    """
    Things the models have no business judging.

    Four categories, because they probe different failure modes and only one
    of them is actually difficult:

    * **People** — `astronaut` and `camera` from scikit-image are photographs
      of human beings. This is the literal case that motivated the guard: a
      webcam pointed at a face, scored as a needle driver at 0.98.
    * **Natural photographs** — a cat, a coffee cup, a rocket, a motorcycle.
      Ordinary images with real-world colour statistics and texture.
    * **Medical, but not endoscopic** — retina fundus, immunohistochemistry,
      a histology slide of skin, a kidney slice, a mitosis field. These are the
      hard negatives and the only ones worth much: they are pink and red
      biological tissue at close range, which is exactly what a surgical frame
      looks like to anything reasoning from colour alone. A guard that rejects
      a cat but accepts a histology slide has learned "is this pink", not "is
      this endoscopic surgery".
    * **Synthetic patterns** — flat colour, noise, gradients, checkerboards,
      plus whatever abstract wallpaper renders the host happens to ship. These
      probe the degenerate end: input with no object structure at all. They are
      the easiest possible negatives and prove the least.

    An earlier version of this function used only macOS desktop pictures and
    described them as landscapes. They are not — on this host they are abstract
    gradient renders, which made the reported separation look like a result
    about natural images when it was a result about smooth colour ramps.

    scikit-image is a development dependency of this script only. Nothing in
    the shipped application imports it.
    """
    frames: list[np.ndarray] = []
    labels: list[str] = []

    def add(image: np.ndarray, label: str) -> None:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.dtype != np.uint8:
            top = float(array.max()) or 1.0
            array = (array.astype(np.float64) / top * 255).astype(np.uint8)
        # Everything downstream treats frames as BGR, the way cv2 reads them.
        frames.append(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
        labels.append(label)

    try:
        from skimage import data as skdata

        for name, kind in [
            ("astronaut", "person"),
            ("camera", "person"),
            ("chelsea", "photo"),
            ("coffee", "photo"),
            ("rocket", "photo"),
            ("stereo_motorcycle", "photo"),
            ("hubble_deep_field", "photo"),
            ("retina", "tissue"),
            ("immunohistochemistry", "tissue"),
            ("skin", "tissue"),
            ("human_mitosis", "tissue"),
            ("cell", "tissue"),
        ]:
            try:
                image = getattr(skdata, name)()
                if isinstance(image, tuple):
                    image = image[0]
                add(image, f"{kind}:{name}")
            except Exception:
                continue

        # 3-D stack; take a middle slice rather than the whole volume.
        try:
            kidney = np.asarray(skdata.kidney())
            add(kidney[kidney.shape[0] // 2], "tissue:kidney")
        except Exception:
            pass
    except ImportError:
        print("   NOTE: scikit-image not installed; skipping photographic probes.")

    # Abstract renders the host happens to ship. Labelled for what they are.
    system_pictures = sorted(Path("/System/Library/Desktop Pictures").glob("*.heic"))[:6]
    if system_pictures:
        with tempfile.TemporaryDirectory() as tmp:
            for source in system_pictures:
                target = Path(tmp) / f"{source.stem}.png"
                # HEIC is not readable by OpenCV; sips ships with macOS.
                result = subprocess.run(
                    ["sips", "-s", "format", "png", "-Z", "1024",
                     str(source), "--out", str(target)],
                    capture_output=True,
                )
                if result.returncode != 0 or not target.exists():
                    continue
                image = cv2.imread(str(target))
                if image is not None:
                    frames.append(image)
                    labels.append(f"abstract:{source.stem}")

    rng = np.random.default_rng(0)
    size = 512
    synthetic = {
        "flat_grey": np.full((size, size, 3), 128, dtype=np.uint8),
        "flat_skin": np.full((size, size, 3), (120, 150, 200), dtype=np.uint8),
        "uniform_noise": rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
        "gradient": np.tile(
            np.linspace(0, 255, size, dtype=np.uint8)[None, :, None], (size, 1, 3)
        ),
        "checkerboard": (
            (np.indices((size, size)).sum(axis=0) // 32 % 2) * 255
        ).astype(np.uint8)[:, :, None].repeat(3, axis=2),
    }
    for name, image in synthetic.items():
        frames.append(image)
        labels.append(f"synthetic:{name}")

    return frames, labels


# ---------------------------------------------------------------------------


def choose_threshold(
    in_distances: np.ndarray,
    out_distances: np.ndarray,
    percentile: float,
    margin: float,
) -> float:
    """
    Put the threshold in the empty band between the two populations.

    The first attempt used `percentile(in, 99) * margin`, which landed at
    0.3555 while real surgical frames reached 0.4334 — so it refused three
    genuine frames to buy nothing, since the nearest probe was at 0.5600. A
    percentile of the in-domain spread is by construction *inside* that spread;
    scaling it by a margin only works if the margin happens to clear the
    maximum, which is luck rather than calibration.

    When the populations separate, the midpoint of the gap is the right answer:
    it is the furthest point from both, so it is the most robust to either
    distribution shifting. The percentile estimate is kept as a floor, so a
    suspiciously tight probe set cannot drag the threshold down below where the
    in-domain data says it should sit.

    When they overlap there is no good threshold, and the honest response is to
    fall back to the percentile and say so rather than pick a number that
    implies a separation the data does not show.
    """
    gap_low = float(in_distances.max())
    gap_high = float(out_distances.min()) if len(out_distances) else float("inf")
    floor = float(np.percentile(in_distances, percentile) * margin)

    if gap_high <= gap_low:
        print(
            f"   WARNING: populations overlap (surgical max {gap_low:.4f} >= "
            f"probe min {gap_high:.4f}). Falling back to the percentile estimate; "
            "the guard will be unreliable in the overlap."
        )
        return floor

    return max(floor, (gap_low + gap_high) / 2.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--percentile", type=float, default=PERCENTILE)
    parser.add_argument("--margin", type=float, default=MARGIN)
    args = parser.parse_args()

    print("1. Exposing the embedding tensor")
    expose_embedding(TOOL_ONNX)

    manifest = json.loads((MODELS / "manifest.json").read_text())
    entry = next(m for m in manifest["models"] if m["name"] == "tool")
    size = entry["imageSize"]

    session = ort.InferenceSession(str(TOOL_ONNX), providers=["CPUExecutionProvider"])
    if EMBEDDING_OUTPUT not in [o.name for o in session.get_outputs()]:
        raise SystemExit("the patched graph does not expose the embedding output")

    print("\n2. Embedding real SurgVU frames")
    surgical = in_domain_frames()
    print(f"   {len(surgical)} frames from {len(json.loads((CLIPS / 'clips.json').read_text()))} clips")
    surgical_vectors = embed(session, surgical, size)

    centroid = surgical_vectors.mean(axis=0)
    centroid /= np.linalg.norm(centroid) or 1.0
    in_distances = 1.0 - surgical_vectors @ centroid

    print("\n3. Embedding out-of-domain probes")
    probes, probe_labels = out_of_domain_frames()
    print(f"   {len(probes)} probes ({sum(1 for p in probe_labels if p.startswith('photo'))} photographs)")
    probe_vectors = embed(session, probes, size)
    out_distances = 1.0 - probe_vectors @ centroid

    threshold = choose_threshold(in_distances, out_distances, args.percentile, args.margin)

    print("\n4. Separation")
    print(f"   surgical   min {in_distances.min():.4f}  median {np.median(in_distances):.4f}  "
          f"p{args.percentile:g} {np.percentile(in_distances, args.percentile):.4f}  max {in_distances.max():.4f}")
    print(f"   probes     min {out_distances.min():.4f}  median {np.median(out_distances):.4f}  "
          f"max {out_distances.max():.4f}")
    print(f"   threshold  {threshold:.4f}")

    false_refusals = int((in_distances > threshold).sum())
    missed = int((out_distances <= threshold).sum())
    print(f"\n   surgical frames refused   : {false_refusals}/{len(in_distances)} "
          f"({false_refusals / max(len(in_distances), 1):.1%})")
    print(f"   probes accepted (missed)  : {missed}/{len(out_distances)}")
    print("\n   per probe category (nearest probe is what matters):")
    by_kind: dict[str, list[float]] = {}
    for label, distance in zip(probe_labels, out_distances):
        by_kind.setdefault(label.split(":")[0], []).append(float(distance))
    for kind in sorted(by_kind, key=lambda k: min(by_kind[k])):
        values = by_kind[kind]
        flag = "  <-- ACCEPTED" if min(values) <= threshold else ""
        print(f"      {kind:<10} n={len(values):<3} min {min(values):.4f}  "
              f"median {float(np.median(values)):.4f}{flag}")

    if missed:
        print()
        for label, distance in zip(probe_labels, out_distances):
            if distance <= threshold:
                print(f"      MISSED {label}  d={distance:.4f}")

    OUTPUT.write_text(
        json.dumps(
            {
                "embedder": "tool.onnx:embedding",
                "dim": int(centroid.shape[0]),
                "imageSize": size,
                "centroid": [round(float(v), 6) for v in centroid],
                "threshold": round(threshold, 6),
                "percentile": args.percentile,
                "margin": args.margin,
                "calibration": {
                    "surgicalFrames": int(len(in_distances)),
                    "surgicalMedian": round(float(np.median(in_distances)), 4),
                    "surgicalMax": round(float(in_distances.max()), 4),
                    "probes": int(len(out_distances)),
                    "probeMin": round(float(out_distances.min()), 4),
                    "probeMedian": round(float(np.median(out_distances)), 4),
                    "falseRefusalRate": round(false_refusals / max(len(in_distances), 1), 4),
                    "probesAccepted": missed,
                    # Per category, because the aggregate hides the only result
                    # that matters: faces and photographs separate cleanly,
                    # non-endoscopic biological tissue does not. Anything
                    # reporting this guard's accuracy must report that too.
                    "byCategory": {
                        kind: {
                            "n": len(values),
                            "min": round(min(values), 4),
                            "median": round(float(np.median(values)), 4),
                            "separates": bool(min(values) > threshold),
                        }
                        for kind, values in sorted(by_kind.items())
                    },
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {OUTPUT.relative_to(REPO)} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
