"""
Grad-CAM re-export for the training pipeline.

The implementation lives in :mod:`ai.explainability.gradcam` so that training
scripts and the live inference server share exactly one code path — an
explanation shown during evaluation and one shown in the dashboard are then
guaranteed to be produced the same way.
"""

from ai.explainability.gradcam import (
    GradCAM,
    find_target_layer,
    heatmap_to_png,
    overlay_heatmap,
)

__all__ = ["GradCAM", "find_target_layer", "overlay_heatmap", "heatmap_to_png"]
