"""
Grad-CAM for surgical models.

Explainability is a first-class feature here, not a debugging aid: every
prediction the platform surfaces can be accompanied by a heatmap showing which
image regions drove it.

Two details matter for correctness across architectures:

**Transformers need reshaping.** A ConvNeXt layer emits ``(B, C, H, W)`` and
can be pooled directly. A ViT/Swin layer emits ``(B, N, C)`` tokens, which must
be reshaped back to a spatial grid (dropping any class token) before pooling,
or the resulting map is meaningless. :meth:`GradCAM._to_spatial` handles both.

**Multi-label needs per-class attribution.** For tool detection, backward is
run from a *single* class logit, so the heatmap answers "where is the evidence
for *this* instrument" rather than blending all twelve.

The implementation registers hooks once and removes them on close, so it is
safe to hold a GradCAM instance for the lifetime of a server process.
"""

from __future__ import annotations

import base64
import io
import logging
import math
from typing import Any, Optional, Sequence

logger = logging.getLogger("charlie.gradcam")


def find_target_layer(model: Any, model_name: str = "") -> Optional[Any]:
    """
    Pick a sensible Grad-CAM target: the last layer that still has spatial
    structure. Falls back to the deepest module that produces a 3-D or 4-D
    activation, which covers architectures not special-cased below.
    """
    backbone = getattr(model, "backbone", model)
    name = (model_name or getattr(model, "model_name", "") or "").lower()

    try:
        if "convnext" in name:
            return backbone.stages[-1]
        if "efficientnet" in name:
            return backbone.conv_head if hasattr(backbone, "conv_head") else backbone.blocks[-1]
        if "resnet" in name:
            return backbone.layer4
        if "swin" in name:
            return backbone.layers[-1].blocks[-1].norm1
        if "vit" in name or "deit" in name or "dino" in name:
            return backbone.blocks[-1].norm1
    except (AttributeError, IndexError):
        logger.debug("Named layer lookup failed for '%s'; scanning modules.", name)

    # Generic: the last module with parameters that is not the classifier.
    candidates = [
        module
        for module in backbone.modules()
        if len(list(module.children())) == 0 and any(p.requires_grad for p in module.parameters(recurse=False))
    ]
    return candidates[-2] if len(candidates) > 1 else (candidates[-1] if candidates else None)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    >>> cam = GradCAM(model, target_layer)          # doctest: +SKIP
    >>> heatmap = cam.generate(tensor, class_idx=3)  # doctest: +SKIP
    >>> cam.close()                                  # doctest: +SKIP
    """

    def __init__(self, model: Any, target_layer: Any, multi_label: bool = True):
        self.model = model
        self.target_layer = target_layer
        self.multi_label = multi_label
        self.activations = None
        self.gradients = None
        self._handles = []
        self._register()

    def _register(self) -> None:
        self._handles.append(self.target_layer.register_forward_hook(self._save_activation))
        self._handles.append(
            self.target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- core ------------------------------------------------------------
    @staticmethod
    def _to_spatial(tensor):
        """
        Normalise an activation/gradient tensor to ``(B, C, H, W)``.

        Transformer blocks emit ``(B, N, C)``; the class token (if present) is
        dropped and the remaining patch tokens are folded back into a square
        grid. Without this, pooling over the token axis produces a heatmap with
        no spatial meaning.
        """
        if tensor.dim() == 4:
            return tensor
        if tensor.dim() == 3:
            batch, tokens, channels = tensor.shape
            side = int(math.sqrt(tokens))
            if side * side != tokens and tokens > 1:
                # Drop a leading class/distillation token.
                tensor = tensor[:, 1:, :]
                tokens = tensor.shape[1]
                side = int(math.sqrt(tokens))
            if side * side != tokens:
                return None
            return tensor.reshape(batch, side, side, channels).permute(0, 3, 1, 2)
        return None

    def generate(
        self,
        input_tensor,
        class_idx: Optional[int] = None,
        upsample_to: Optional[tuple[int, int]] = None,
    ):
        """
        Produce a normalised ``(H, W)`` heatmap in ``[0, 1]``.

        Returns ``None`` when the target layer produces activations Grad-CAM
        cannot interpret, rather than a misleading blank map.
        """
        import torch

        self.model.zero_grad(set_to_none=True)

        # Gradients are required, so inference_mode/no_grad must not be active.
        with torch.enable_grad():
            output = self.model(input_tensor)

            if output.dim() == 1:
                output = output.unsqueeze(0)

            if class_idx is None:
                class_idx = int(output[0].argmax().item())

            score = output[0, class_idx]
            score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            logger.warning("Grad-CAM hooks captured nothing for this layer.")
            return None

        activations = self._to_spatial(self.activations.detach())
        gradients = self._to_spatial(self.gradients.detach())
        if activations is None or gradients is None:
            logger.warning(
                "Target layer produced a shape Grad-CAM cannot map spatially (%s).",
                tuple(self.activations.shape),
            )
            return None

        # Channel importance = global-average-pooled gradient.
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        size = upsample_to or (input_tensor.shape[-2], input_tensor.shape[-1])
        cam = torch.nn.functional.interpolate(
            cam, size=size, mode="bilinear", align_corners=False
        )

        cam = cam[0, 0]
        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) < 1e-8:
            # A uniform map means the class had no localised evidence.
            return cam.zero_().cpu().numpy() if hasattr(cam, "cpu") else None
        cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.cpu().numpy()

    def generate_batch(
        self, input_tensor, class_indices: Sequence[int]
    ) -> list:
        """One heatmap per requested class, for side-by-side comparison."""
        return [self.generate(input_tensor, class_idx=idx) for idx in class_indices]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _jet(value: float) -> tuple[int, int, int]:
    """Jet-like colormap, pure Python so no OpenCV dependency is required."""
    value = min(max(value, 0.0), 1.0)
    four = 4.0 * value
    red = min(four - 1.5, -four + 4.5)
    green = min(four - 0.5, -four + 3.5)
    blue = min(four + 0.5, -four + 2.5)
    clamp = lambda c: int(255 * min(max(c, 0.0), 1.0))  # noqa: E731
    return clamp(red), clamp(green), clamp(blue)


def overlay_heatmap(image, heatmap, alpha: float = 0.5):
    """
    Blend a heatmap over a PIL image. Returns a PIL image.

    ``alpha`` is modulated by heatmap intensity, so cold regions stay legible
    instead of being washed out by a flat colour wash — you can still see the
    anatomy under the explanation.
    """
    from PIL import Image

    image = image.convert("RGB")
    width, height = image.size

    rows = len(heatmap)
    cols = len(heatmap[0]) if rows else 0
    if not rows or not cols:
        return image

    coloured = Image.new("RGB", (cols, rows))
    mask = Image.new("L", (cols, rows))
    coloured_px = coloured.load()
    mask_px = mask.load()

    for y in range(rows):
        row = heatmap[y]
        for x in range(cols):
            value = float(row[x])
            coloured_px[x, y] = _jet(value)
            mask_px[x, y] = int(255 * alpha * value)

    coloured = coloured.resize((width, height), Image.BILINEAR)
    mask = mask.resize((width, height), Image.BILINEAR)

    result = image.copy()
    result.paste(coloured, (0, 0), mask)
    return result


def heatmap_to_png(image, heatmap, alpha: float = 0.5) -> Optional[str]:
    """Render an overlay and return it as a ``data:image/png;base64`` URL."""
    try:
        overlay = overlay_heatmap(image, heatmap, alpha=alpha)
        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:
        logger.error("Could not render Grad-CAM overlay: %s", exc)
        return None
