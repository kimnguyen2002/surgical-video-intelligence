"""
Export a trained checkpoint to ONNX for accelerated serving.

    python -m ai.training.export_onnx --checkpoint checkpoints/tool_best.pt --task tool

ONNX Runtime typically gives a solid CPU speedup over eager PyTorch and opens
the door to TensorRT on NVIDIA hardware. The exported graph uses a dynamic
batch axis so the server can batch adaptively under load.

Note the trade-off: an ONNX graph has no autograd, so Grad-CAM cannot run on
it. Deployments that want both keep the PyTorch model for explainability and
use ONNX for bulk inference — which is why this is an export step rather than
a replacement.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from ai.training.config import config
from ai.training.models import load_checkpoint
from ai.training.train_tool_detection import resolve_device

logging.basicConfig(level=logging.INFO, format="%(levelname)-5s │ %(message)s")
logger = logging.getLogger("surgvu.export")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a SurgVU checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=["tool", "step"], default="tool")
    parser.add_argument("--output", default="")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--clip-len", type=int, default=config.clip_length)
    parser.add_argument("--verify", action="store_true", help="Compare ONNX vs PyTorch outputs.")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        logger.error("Checkpoint not found: %s", checkpoint)
        return 1

    device = resolve_device("cpu")  # Export from CPU for a portable graph.
    model, state = load_checkpoint(checkpoint, task=args.task, device=str(device))
    model.eval()

    size = state.get("image_size", config.image_size)
    if args.task == "tool":
        dummy = torch.randn(1, 3, size, size)
        dynamic = {"input": {0: "batch"}, "logits": {0: "batch"}}
    else:
        clip_length = state.get("clip_length", args.clip_len)
        dummy = torch.randn(1, clip_length, 3, size, size)
        dynamic = {"input": {0: "batch"}, "logits": {0: "batch"}}

    output = Path(args.output) if args.output else checkpoint.with_suffix(".onnx")

    logger.info("Exporting %s → %s (opset %d)", checkpoint.name, output.name, args.opset)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(output),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=dynamic,
            opset_version=args.opset,
            do_constant_folding=True,
        )
    except Exception as exc:
        logger.error(
            "ONNX export failed: %s\n"
            "Some timm backbones use ops without an ONNX mapping at this opset. "
            "Try --opset 18, or export a ConvNeXt/EfficientNet variant.",
            exc,
        )
        return 1

    size_mb = output.stat().st_size / 1024**2
    logger.info("Exported %.1f MB → %s", size_mb, output)

    if args.verify:
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError:
            logger.warning("Verification needs onnxruntime: pip install onnxruntime")
            return 0

        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        onnx_out = session.run(None, {"input": dummy.numpy()})[0]
        with torch.no_grad():
            torch_out = model(dummy).numpy()

        difference = float(np.abs(onnx_out - torch_out).max())
        logger.info("Max |ONNX − PyTorch| = %.2e", difference)
        if difference > 1e-3:
            logger.warning("Outputs diverge more than expected — inspect before serving.")
        else:
            logger.info("✅ Numerically equivalent.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
