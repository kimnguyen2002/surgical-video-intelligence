#!/usr/bin/env python3
"""
Export the trained SurgVU checkpoints to ONNX for in-browser inference.

Why this exists: calling Gemini once per frame costs a network round trip
(hundreds of milliseconds at best) and burns a free-tier request each time —
20 of them and the key is rate-limited. Neither is survivable for continuous
intraoperative detection. The models in `ml/checkpoints/` were
trained on this exact dataset and are small enough to run in the page, where a
frame never leaves the machine and there is no quota to exhaust.

Three checkpoints, three jobs:

    tool_best.pt       ConvNeXtV2-Atto, 224px, 14 classes, multi-label sigmoid
                       "which instruments are in this frame"
    task_best.pt       ConvNeXtV2-Atto, 176px, 8 classes, softmax
                       "which surgical task is happening"
    detection_best.pt  YOLO, boxes + classes
                       "where in the frame each instrument is"

Run from the repository root:

    python3 scripts/export_models.py

Outputs land in `public/models/`, which Vite serves as static assets.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# The training package and the PyTorch checkpoints both live in this repository
# now, under ml/. They previously sat in a sibling `surgical_main/` directory,
# which meant this script only ran on the one machine that had it — and a
# checkpoint whose producing code is not beside it is an unverifiable artefact.
MAIN = REPO / "ml"
CHECKPOINTS = MAIN / "checkpoints"
OUT_DIR = REPO / "public" / "models"

# `ml/training/...` is imported as `ai.training...` by the training code, so the
# package is reached through a path entry rather than renamed imports.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402


def fold_external_data(path: Path) -> None:
    """Rewrite an ONNX file so all weights live inside it."""
    import onnx

    model = onnx.load(str(path), load_external_data=True)
    onnx.save(model, str(path), save_as_external_data=False)
    for sidecar in path.parent.glob(f"{path.name}.data"):
        sidecar.unlink()


def quantize(path: Path) -> Path:
    """
    Dynamic int8 quantization of the weights.

    Roughly quarters the download and speeds up the WASM backend, which is the
    one that runs when a machine has no WebGPU. Activations stay float, so the
    accuracy cost is far smaller than full static quantization — but it is not
    zero, which is why the caller measures it against the float graph rather
    than assuming.
    """
    import tempfile

    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    out = path.with_name(path.stem + ".int8.onnx")

    # torch's dynamo exporter leaves value_info entries the quantizer's shape
    # inference disagrees with ("Inferred shape and existing shape differ").
    # quant_pre_process re-runs symbolic shape inference and folds the
    # constants, which reconciles them.
    with tempfile.TemporaryDirectory() as tmp:
        prepared = Path(tmp) / "prepared.onnx"
        quant_pre_process(str(path), str(prepared), skip_symbolic_shape=False)
        quantize_dynamic(str(prepared), str(out), weight_type=QuantType.QUInt8)
    return out


def export_classifier(name: str, checkpoint: Path, opset: int) -> dict:
    """
    Export one of the two ConvNeXtV2 classifiers.

    The checkpoint stores fp16 weights to keep the repository small. They are
    widened to fp32 before export, the same way `models.load_checkpoint` does —
    a graph left holding half-precision parameters fails on its first forward
    pass rather than at load time.
    """
    from ai.training.models import build_model

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    size = int(state.get("image_size", 224))
    classes = list(state.get("classes", []))

    model = build_model(
        "tool",
        model_name=state["model_name"],
        num_classes=int(state["num_classes"]),
        pretrained=False,
    )
    weights = {
        k: (v.float() if getattr(v, "dtype", None) == torch.float16 else v)
        for k, v in state["model"].items()
    }
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        print(f"  ! {len(missing)} missing / {len(unexpected)} unexpected keys")
    model.eval()

    dummy = torch.randn(1, 3, size, size)
    with torch.no_grad():
        reference = model(dummy)

    out = OUT_DIR / f"{name}.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["logits"],
        # Batch stays fixed at 1: the page analyses one frame at a time, and a
        # static shape lets onnxruntime-web pre-plan every allocation instead
        # of re-planning on each call.
        dynamic_axes=None,
        opset_version=opset,
        do_constant_folding=True,
        # The TorchScript exporter rather than the dynamo one. Dynamo emits
        # value_info the int8 quantizer's shape inference cannot reconcile
        # (ReduceMean with no axes attribute), and this graph is a plain CNN
        # with nothing the older path handles badly.
        dynamo=False,
    )

    # torch.onnx.export spills weights above a threshold into a sidecar
    # `.onnx.data` file. onnxruntime-web can be taught to fetch those, but it
    # turns one static asset into two that must stay adjacent and be uploaded
    # together — so they are folded back into a single self-contained graph.
    fold_external_data(out)

    # An export that silently diverges from the checkpoint is worse than one
    # that fails, so the graph is run and compared before it ships.
    import onnxruntime as ort

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]
    delta = float(abs(onnx_out - reference.numpy()).max())
    print(f"  max |onnx - torch| = {delta:.3e}")
    if delta > 1e-3:
        raise SystemExit(f"{name}: ONNX output diverges from PyTorch by {delta}")

    quantized = quantize(out)
    q_session = ort.InferenceSession(str(quantized), providers=["CPUExecutionProvider"])
    q_out = q_session.run(None, {"input": dummy.numpy()})[0]
    q_delta = float(abs(q_out - reference.numpy()).max())
    print(f"  int8: {quantized.stat().st_size / 1e6:.1f} MB, max |int8 - torch| = {q_delta:.3e}")

    return {
        "name": name,
        "file": out.name,
        "int8File": quantized.name,
        "int8SizeBytes": quantized.stat().st_size,
        "int8MaxDelta": q_delta,
        "imageSize": size,
        "classes": classes,
        "numClasses": int(state["num_classes"]),
        "activation": "sigmoid" if state.get("task") == "tool_presence" else "softmax",
        "task": state.get("task", ""),
        "metric": {
            k: state[k]
            for k in ("val_mAP", "val_balanced_accuracy", "val_accuracy")
            if k in state
        },
        "provenance": state.get("provenance_note", ""),
        "sizeBytes": out.stat().st_size,
        "maxExportDelta": delta,
    }


def export_detector(checkpoint: Path, imgsz: int, opset: int) -> dict:
    """
    Export the YOLO instrument detector.

    Ultralytics writes the ONNX next to the .pt, so it is moved into place
    afterwards. `simplify` folds the constant subgraphs that the WASM backend
    would otherwise re-evaluate on every frame.
    """
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    names = model.names
    exported = model.export(
        format="onnx", imgsz=imgsz, simplify=True, opset=opset, dynamic=False, half=False
    )

    out = OUT_DIR / "detection.onnx"
    shutil.move(str(exported), out)
    fold_external_data(out)

    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    shape = session.get_outputs()[0].shape
    print(f"  output shape {shape}")

    probe = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    reference = session.run(None, {session.get_inputs()[0].name: probe})[0]

    quantized = quantize(out)
    q_session = ort.InferenceSession(str(quantized), providers=["CPUExecutionProvider"])
    q_out = q_session.run(None, {q_session.get_inputs()[0].name: probe})[0]
    q_delta = float(abs(q_out - reference).max())
    print(f"  int8: {quantized.stat().st_size / 1e6:.1f} MB, max |int8 - float| = {q_delta:.3e}")

    return {
        "name": "detection",
        "file": out.name,
        "int8File": quantized.name,
        "int8SizeBytes": quantized.stat().st_size,
        "int8MaxDelta": q_delta,
        "inputName": session.get_inputs()[0].name,
        "outputName": session.get_outputs()[0].name,
        "imageSize": imgsz,
        "classes": [names[i] for i in sorted(names)],
        "numClasses": len(names),
        "outputShape": [str(d) for d in shape],
        "sizeBytes": out.stat().st_size,
    }


def reapply_graph_patches(skip: bool = False) -> None:
    """
    Re-add the two extra outputs this export just discarded.

    `tool.onnx` carries two graph outputs beyond `logits`, added after the
    fact by other scripts: `embedding`, which the out-of-domain guard measures
    distance in, and `features`, which the activation heatmap is computed from.
    A fresh export overwrites the file and takes both with it.

    The failure that causes is quiet. The worker treats a missing output as
    "that capability did not ship" and carries on, so the app keeps detecting
    instruments while the guard stops refusing anything and the heatmap button
    disappears. Nothing errors; two safety-relevant features just stop
    existing. Re-running the patches here means they cannot be forgotten.

    Both patch scripts are idempotent, so this is safe to run at any time.
    """
    if skip:
        print("\n(skipping graph patches; run export_cam.py and "
              "build_domain_guard.py yourself before shipping)")
        return

    print("\nRe-applying graph patches that this export overwrote:")
    for module_name, label in (
        ("export_cam", "activation heatmap (features output + CAM weights)"),
        ("build_domain_guard", "out-of-domain guard (embedding output + reference)"),
    ):
        print(f"  - {label}")
        try:
            module = __import__(module_name)
            if module.main() != 0:
                print(f"    FAILED: {module_name} returned non-zero")
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {module_name}: {exc}")
            print("    Run it by hand before shipping; the capability is "
                  "silently absent until you do.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--det-size",
        type=int,
        default=448,
        help="YOLO input size. 640 is native; smaller trades a little recall "
        "for a lot of latency, which is the right trade for a live loop.",
    )
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="skip re-adding the embedding/features graph outputs afterwards",
    )
    args = parser.parse_args()

    if not CHECKPOINTS.is_dir():
        print(f"Checkpoints not found at {CHECKPOINTS}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"models": [], "opset": args.opset}

    print("Exporting tool presence classifier...")
    manifest["models"].append(export_classifier("tool", CHECKPOINTS / "tool_best.pt", args.opset))

    print("Exporting surgical task classifier...")
    manifest["models"].append(export_classifier("task", CHECKPOINTS / "task_best.pt", args.opset))

    print("Exporting instrument detector...")
    manifest["models"].append(export_detector(CHECKPOINTS / "detection_best.pt", args.det_size, args.opset))

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = sum(m["sizeBytes"] for m in manifest["models"])
    print(f"\nWrote {len(manifest['models'])} models to {OUT_DIR} ({total / 1e6:.1f} MB total)")
    for m in manifest["models"]:
        print(f"  {m['file']:<18} {m['sizeBytes'] / 1e6:>6.1f} MB  {m['numClasses']} classes @ {m['imageSize']}px")

    reapply_graph_patches(skip=args.no_patch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
