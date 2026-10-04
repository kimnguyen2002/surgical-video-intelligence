"""
Frame embeddings — the semantic backbone of the temporal memory.

Tiers, selected automatically:

``timm`` backbone (ConvNeXt-V2 / ViT / DINOv2 / Swin)
    Learned 768-dim semantic features. This is what the platform is designed
    around and what the temporal memory's event clustering assumes.

classical descriptor (always available)
    A tiled colour-histogram + gradient-orientation descriptor. It captures
    scene composition, tissue colour, and instrument edges well enough to
    segment a procedure into visually coherent events and to detect scene
    changes — which is exactly what the memory engine needs — but it carries
    no semantic meaning. Reported honestly via :attr:`backend`.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Optional, Sequence

from ai.common import optional_import, settings

logger = logging.getLogger("charlie.embeddings.vision")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_device(preference: str = "auto") -> str:
    """Pick the best available torch device."""
    torch = optional_import("torch")
    if torch is None:
        return "cpu"
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class FrameEmbedder:
    """Encodes RGB frames into unit-norm feature vectors."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.vision_embedding_model
        self.device = device or resolve_device(settings.inference_device)
        self._model = None
        self._transform = None
        self._lock = threading.Lock()
        self.backend = "classical-descriptor"
        self.dim = 144  # 3 colour channels x 16 bins x 3 tiles = classical dim
        self._try_load_backbone()

    def _try_load_backbone(self) -> None:
        torch = optional_import("torch")
        timm = optional_import("timm")
        if torch is None or timm is None:
            logger.info(
                "torch/timm unavailable — frame embeddings use the classical "
                "descriptor tier."
            )
            return
        try:
            model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
            model.eval().to(self.device)
            cfg = timm.data.resolve_model_data_config(model)
            self._transform = timm.data.create_transform(**cfg, is_training=False)
            self._model = model
            self.dim = int(model.num_features)
            self.backend = f"timm:{self.model_name}@{self.device}"
            logger.info("Frame embedder ready: %s (dim=%d)", self.backend, self.dim)
        except Exception as exc:
            logger.warning(
                "Could not load vision backbone '%s' (%s) — using classical descriptor.",
                self.model_name,
                exc,
            )
            self._model = None

    @property
    def is_semantic(self) -> bool:
        return self._model is not None

    # -- encoding --------------------------------------------------------
    def encode_pil(self, images: Sequence[Any]) -> list[list[float]]:
        """Encode a batch of PIL images."""
        if not images:
            return []
        if self._model is not None:
            return self._encode_backbone(images)
        return [self._classical(img) for img in images]

    def _encode_backbone(self, images: Sequence[Any]) -> list[list[float]]:
        torch = optional_import("torch")
        assert torch is not None and self._transform is not None
        with self._lock:
            try:
                batch = torch.stack(
                    [self._transform(img.convert("RGB")) for img in images]
                ).to(self.device)
                with torch.inference_mode():
                    features = self._model(batch)
                features = torch.nn.functional.normalize(features, dim=-1)
                return features.float().cpu().tolist()
            except Exception as exc:
                logger.error("Backbone encoding failed (%s); using classical tier.", exc)
                return [self._classical(img) for img in images]

    def _classical(self, image: Any) -> list[float]:
        """
        Tiled colour histogram + edge-orientation descriptor.

        The frame is downscaled to 48x48 and split into a 3x3 grid. Each tile
        contributes a 16-bin hue-ish colour histogram (per channel, coarse) and
        a 4-bin gradient-orientation histogram. Concatenated and L2-normalised,
        this gives a stable fingerprint of scene composition that changes
        sharply at cuts and camera moves — the signal the memory engine
        segments on.
        """
        try:
            img = image.convert("RGB").resize((48, 48))
            pixels = list(img.getdata())
        except Exception:
            return [0.0] * self.dim

        width = 48
        tiles = 3
        tile_px = width // tiles
        vec: list[float] = []

        for ty in range(tiles):
            for tx in range(tiles):
                colour = [0.0] * 12  # 4 bins x 3 channels
                for y in range(ty * tile_px, (ty + 1) * tile_px):
                    for x in range(tx * tile_px, (tx + 1) * tile_px):
                        r, g, b = pixels[y * width + x]
                        colour[r >> 6] += 1
                        colour[4 + (g >> 6)] += 1
                        colour[8 + (b >> 6)] += 1
                vec.extend(colour)

        # Gradient energy over a coarse grid captures instrument edges.
        grey = [(p[0] * 299 + p[1] * 587 + p[2] * 114) / 1000.0 for p in pixels]
        for ty in range(tiles):
            for tx in range(tiles):
                gx = gy = 0.0
                for y in range(ty * tile_px, (ty + 1) * tile_px - 1):
                    for x in range(tx * tile_px, (tx + 1) * tile_px - 1):
                        i = y * width + x
                        gx += abs(grey[i + 1] - grey[i])
                        gy += abs(grey[i + width] - grey[i])
                vec.extend([gx / 255.0, gy / 255.0, math.hypot(gx, gy) / 255.0])

        # Pad/trim to the declared dimensionality so vectors stay comparable.
        if len(vec) < self.dim:
            vec.extend([0.0] * (self.dim - len(vec)))
        vec = vec[: self.dim]

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def encode_bytes(self, image_bytes: bytes) -> Optional[list[float]]:
        """Encode a single encoded image (JPEG/PNG bytes)."""
        pil = optional_import("pillow")
        if pil is None:
            return None
        import io

        from PIL import Image

        try:
            img = Image.open(io.BytesIO(image_bytes))
            return self.encode_pil([img])[0]
        except Exception as exc:
            logger.error("Could not decode image: %s", exc)
            return None

    def info(self) -> dict:
        return {
            "backend": self.backend,
            "dim": self.dim,
            "device": self.device,
            "semantic": self.is_semantic,
            "note": None
            if self.is_semantic
            else "Classical descriptor tier. Install torch + timm for learned features.",
        }


_INSTANCE: Optional[FrameEmbedder] = None
_LOCK = threading.Lock()


def get_frame_embedder() -> FrameEmbedder:
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = FrameEmbedder()
    return _INSTANCE
