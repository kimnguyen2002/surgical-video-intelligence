"""
Run a trained SurgVU model on an image or a video.

    python -m ai.training.predict --image frame.jpg --checkpoint checkpoints/tool_best.pt
    python -m ai.training.predict --video case.mp4 --checkpoint checkpoints/tool_best.pt \
        --interval 2 --output timeline.json

For a single image this prints per-class probabilities and writes a Grad-CAM
overlay. For a video it samples frames at a fixed interval, runs tool
detection, and emits a JSON timeline — the same shape the platform's temporal
memory ingests, so an offline run and a live session produce comparable output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from ai.training.config import config
from ai.training.models import load_checkpoint
from ai.training.train_tool_detection import resolve_device

logging.basicConfig(level=logging.INFO, format="%(levelname)-5s │ %(message)s")
logger = logging.getLogger("surgvu.predict")


def build_transform():
    import torchvision.transforms as T

    return T.Compose(
        [
            T.Resize(int(config.image_size * 1.14)),
            T.CenterCrop(config.image_size),
            T.ToTensor(),
            T.Normalize(mean=config.mean, std=config.std),
        ]
    )


def predict_image(model, image, transform, classes, device, threshold: float) -> dict:
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(tensor))[0].float().cpu().tolist()

    tools = [
        {
            "name": name,
            "display": config.tool_display(name),
            "confidence": round(probability, 4),
            "present": probability >= threshold,
        }
        for name, probability in zip(classes, probabilities)
    ]
    tools.sort(key=lambda t: -t["confidence"])
    return {"tools": tools, "tensor": tensor}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SurgVU inference.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--video")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=config.tool_threshold)
    parser.add_argument("--interval", type=float, default=2.0, help="Video sampling interval (s).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="", help="Where to write JSON results.")
    parser.add_argument("--no-gradcam", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        logger.error(
            "Checkpoint not found: %s\nTrain one with:\n"
            "  python -m ai.training.train_tool_detection",
            checkpoint,
        )
        return 1

    device = resolve_device(args.device)
    model, state = load_checkpoint(checkpoint, task="tool", device=str(device))
    classes = state.get("classes", config.tool_classes)
    transform = build_transform()

    from PIL import Image

    if args.image:
        path = Path(args.image)
        if not path.exists():
            logger.error("Image not found: %s", path)
            return 1

        image = Image.open(path)
        result = predict_image(model, image, transform, classes, device, args.threshold)

        print(f"\nTool predictions for {path.name}:")
        for tool in result["tools"]:
            mark = "✓" if tool["present"] else " "
            print(f"  [{mark}] {tool['display']:<34} {tool['confidence']:.3f}")

        payload = {"image": str(path), "tools": result["tools"]}

        if not args.no_gradcam:
            from ai.explainability import GradCAM, heatmap_to_png

            layer = model.gradcam_layer()
            if layer is not None:
                top = classes.index(result["tools"][0]["name"])
                with GradCAM(model, layer) as cam:
                    heatmap = cam.generate(result["tensor"], class_idx=top)
                if heatmap is not None:
                    data_url = heatmap_to_png(image.convert("RGB"), heatmap, alpha=0.55)
                    if data_url:
                        import base64

                        out = path.with_name(f"{path.stem}_gradcam.png")
                        out.write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
                        payload["gradcam"] = str(out)
                        print(f"\nGrad-CAM for '{result['tools'][0]['display']}' → {out}")

        output = Path(args.output) if args.output else path.with_suffix(".predictions.json")
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results → {output}")
        return 0

    # -- video -----------------------------------------------------------
    try:
        import cv2
    except ImportError:
        logger.error("Video inference needs OpenCV: pip install opencv-python-headless")
        return 1

    path = Path(args.video)
    if not path.exists():
        logger.error("Video not found: %s", path)
        return 1

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        logger.error("Could not open %s", path)
        return 1

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * args.interval)))
    timeline = []
    index = 0

    logger.info("Sampling every %.1fs (%d frames) from %s", args.interval, step, path.name)
    while True:
        ok = capture.grab()
        if not ok:
            break
        if index % step == 0:
            ok, frame = capture.retrieve()
            if ok:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = predict_image(model, image, transform, classes, device, args.threshold)
                present = [t["name"] for t in result["tools"] if t["present"]]
                timeline.append(
                    {
                        "timestamp": round(index / fps, 2),
                        "tools": present,
                        "confidences": {
                            t["name"]: t["confidence"] for t in result["tools"][:5]
                        },
                    }
                )
                if len(timeline) % 25 == 0:
                    logger.info("  %d samples…", len(timeline))
        index += 1
    capture.release()

    payload = {
        "video": str(path),
        "checkpoint": str(checkpoint),
        "interval": args.interval,
        "samples": len(timeline),
        "timeline": timeline,
    }
    output = Path(args.output) if args.output else path.with_suffix(".timeline.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %d samples → %s", len(timeline), output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
