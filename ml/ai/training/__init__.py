"""
SurgVU deep learning pipeline — arXiv:2501.09209v1.

Two tasks, matching the dataset's two label files:

* **Tool detection** — multi-label frame classification over 12 robotic
  instruments (``tools.csv``).
* **Step recognition** — temporal classification over 8 surgical tasks
  (``tasks.csv``).

Run everything as modules from the repository root so ``ai.*`` imports
resolve::

    python -m ai.training.download_dataset --dest data/surgvu --extract
    python -m ai.training.extract_frames --dataset data/surgvu --fps 1
    python -m ai.training.train_tool_detection --model convnext_tiny
    python -m ai.training.train_step_recognition --model cnn_gru
    python -m ai.training.evaluate --task tool --checkpoint checkpoints/tool_best.pt
"""

from .config import TrainingConfig, config

__all__ = ["TrainingConfig", "config"]
