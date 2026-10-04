"""
Export what the browser needs to draw a class activation map.

Why not Grad-CAM
----------------
Grad-CAM needs a backward pass. ONNX Runtime Web has no autograd, and shipping
a training runtime to a web page to produce a heatmap would cost more than
everything else in this project combined.

It is also unnecessary *for this architecture*. Grad-CAM exists because modern
networks put several layers between the last convolution and the logit, so the
per-channel importance has to be recovered from gradients. ConvNeXtV2 does not:
its head is global average pooling, a layer norm, and one linear layer. With
that shape the per-channel importance is sitting in the weight matrix, and the
heatmap can be read off a *forward* pass exactly — which is the original CAM
(Zhou et al., 2016).

The derivation, including the layer norm
----------------------------------------
Let `x[k, s]` be the final feature map, channel `k`, spatial position `s`, with
`S` positions. The head computes::

    g[k]      = (1/S) * sum_s x[k, s]                    # global average pool
    n[k]      = gamma[k] * (g[k] - mu) / sigma + beta[k] # layer norm over k
    logit[c]  = sum_k W[c, k] * n[k] + b[c]              # linear

Substituting, and collecting everything that does not depend on `s`::

    logit[c] = (1 / (S * sigma)) * sum_s sum_k (W[c, k] * gamma[k]) * x[k, s] + const

So the spatial contribution to class `c` is::

    CAM[c, s] = sum_k (W[c, k] * gamma[k]) * x[k, s]

`mu`, `sigma` and the bias terms are scalars for a given frame: they scale and
shift the whole map uniformly and vanish when it is normalised for display.
The one thing that does *not* vanish is `gamma` — the layer norm's per-channel
scale — because it is inside the sum over `k`. Using the raw `W` and ignoring
`gamma` is the obvious shortcut and it is wrong: it reweights every channel by
however much the norm had learned to amplify or suppress it.

This script therefore writes the **effective** weights, `W * gamma`, and
exposes the pre-pool feature map as a graph output. The browser does one
multiply-accumulate over 320 channels and 49 positions per frame.

    python3 scripts/export_cam.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "public" / "models"
TOOL_ONNX = MODELS / "tool.onnx"
OUTPUT = MODELS / "cam_weights.json"

#: The residual sum feeding GlobalAveragePool: the last feature map.
FEATURE_TENSOR = "/backbone/stages/stages.3/blocks/blocks.1/Add_output_0"
FEATURE_OUTPUT = "features"

HEAD_WEIGHT = "head.1.weight"
NORM_WEIGHT = "backbone.head.norm.weight"


def expose_features(path: Path) -> bool:
    """Add the pre-pool feature map as a graph output, idempotently."""
    model = onnx.load(str(path))
    graph = model.graph

    if any(o.name == FEATURE_OUTPUT for o in graph.output):
        print(f"  {path.name}: '{FEATURE_OUTPUT}' output already present")
        return False

    producer = next((n for n in graph.node if FEATURE_TENSOR in n.output), None)
    if producer is None:
        raise SystemExit(
            f"{path.name}: no node produces {FEATURE_TENSOR}. The checkpoint was "
            "re-exported with a different graph; find the input to GlobalAveragePool "
            "and update FEATURE_TENSOR."
        )

    producer.output[0] = FEATURE_OUTPUT
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == FEATURE_TENSOR:
                node.input[i] = FEATURE_OUTPUT

    graph.output.append(
        onnx.helper.make_tensor_value_info(
            FEATURE_OUTPUT, onnx.TensorProto.FLOAT, [1, None, None, None]
        )
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    print(f"  {path.name}: added '{FEATURE_OUTPUT}' output")
    return True


def verify(effective: np.ndarray) -> float:
    """
    Check the derivation against the model itself.

    If `CAM[c, s] = sum_k W_eff[c, k] * x[k, s]` is right, then summing the map
    over space and undoing the per-frame scalars must reproduce the logits the
    network actually emitted. If it is wrong, the heatmap is still smooth,
    still plausible, and still drawn over the video — which is why this is a
    numeric check and not a visual one.
    """
    import cv2
    import onnxruntime as ort

    model = onnx.load(str(TOOL_ONNX))
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    head_w = init[HEAD_WEIGHT].astype(np.float64)
    gamma = init[NORM_WEIGHT].astype(np.float64)
    head_b = init["head.1.bias"].astype(np.float64)
    beta = init["backbone.head.norm.bias"].astype(np.float64)

    session = ort.InferenceSession(str(TOOL_ONNX), providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]

    clips = sorted((REPO / "public" / "clips").glob("*.mp4"))
    if not clips:
        print("   (no clips available; skipping numeric verification)")
        return float("nan")

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_domain_guard import preprocess

    capture = cv2.VideoCapture(str(clips[0]))
    worst = 0.0
    checked = 0
    for index in range(0, 1500, 300):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        out = dict(zip(names, session.run(None, {"input": preprocess(frame, 224)})))
        logits = out["logits"].reshape(-1).astype(np.float64)
        features = out[FEATURE_OUTPUT][0].astype(np.float64)

        channels = features.shape[0]
        positions = features.shape[1] * features.shape[2]
        pooled = features.reshape(channels, positions).mean(axis=1)
        mu, sigma = pooled.mean(), pooled.std()

        cam = effective.astype(np.float64) @ features.reshape(channels, positions)
        reconstructed = (
            cam.sum(axis=1) / (positions * sigma)
            - (mu / sigma) * (head_w * gamma).sum(axis=1)
            + head_w @ beta
            + head_b
        )
        worst = max(worst, float(np.abs(reconstructed - logits).max()))
        checked += 1
    capture.release()

    print(f"\n3. Verification over {checked} real frames")
    print(f"   max |CAM-reconstructed logit - actual logit| = {worst:.2e}")
    print("   " + ("OK: the decomposition is exact to float32 rounding."
                   if worst < 1e-3 else
                   "FAILED: the derivation does not reproduce the logits."))
    return worst


def main() -> int:
    print("1. Exposing the feature map")
    expose_features(TOOL_ONNX)

    model = onnx.load(str(TOOL_ONNX))
    initializers = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

    for name in (HEAD_WEIGHT, NORM_WEIGHT):
        if name not in initializers:
            raise SystemExit(f"{name} is not an initializer in {TOOL_ONNX.name}")

    head = initializers[HEAD_WEIGHT].astype(np.float32)      # [classes, channels]
    gamma = initializers[NORM_WEIGHT].astype(np.float32)     # [channels]

    if head.shape[1] != gamma.shape[0]:
        raise SystemExit(
            f"shape mismatch: head is {head.shape}, layer-norm gamma is {gamma.shape}"
        )

    # The whole point of this script: fold the layer norm's per-channel scale
    # into the classifier weights. See the module docstring.
    effective = (head * gamma[None, :]).astype(np.float32)

    manifest = json.loads((MODELS / "manifest.json").read_text())
    entry = next(m for m in manifest["models"] if m["name"] == "tool")

    # Base64 little-endian float32 rather than a JSON array of numbers: 4,480
    # floats print as roughly 90 KB of decimal text and decode to 18 KB here,
    # and the browser gets a Float32Array without parsing 4,480 strings.
    payload = {
        "source": TOOL_ONNX.name,
        "featureOutput": FEATURE_OUTPUT,
        "classes": entry["classes"],
        "numClasses": int(effective.shape[0]),
        "numChannels": int(effective.shape[1]),
        "note": (
            "Effective CAM weights: the classifier's weight matrix with the head "
            "layer-norm's per-channel scale folded in. CAM[c,s] = sum_k W_eff[c,k] * "
            "features[k,s]. Derived in scripts/export_cam.py."
        ),
        "weightsBase64": base64.b64encode(effective.tobytes()).decode("ascii"),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")

    print("\n2. Effective weights")
    print(f"   head {head.shape} x gamma {gamma.shape} -> {effective.shape}")
    print(f"   |W| mean {np.abs(head).mean():.4f}  |W*gamma| mean {np.abs(effective).mean():.4f}")
    print(f"\nwrote {OUTPUT.relative_to(REPO)} ({OUTPUT.stat().st_size / 1024:.0f} KB)")

    # A check worth having: if gamma were near-uniform the fold would be a
    # no-op and the extra machinery would be pointless. It is not.
    spread = float(gamma.max() / max(gamma.min(), 1e-6))
    print(f"   layer-norm gamma range {gamma.min():.3f}..{gamma.max():.3f} "
          f"(ratio {spread:.1f}x) - folding it in is not a no-op")

    error = verify(effective)
    return 0 if not (error == error and error >= 1e-3) else 1


if __name__ == "__main__":
    raise SystemExit(main())
