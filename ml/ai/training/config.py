"""
Configuration for the SurgVU training pipeline.

Class names are the dataset's own, normalised to snake_case identifiers. The
display names are kept alongside so reports and the UI can print what the
paper prints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(key: str, default):
    raw = os.environ.get(key)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(raw)
        except ValueError:
            return default
    return raw


# ---------------------------------------------------------------------------
# Class vocabularies (SurgVU, arXiv:2501.09209v1)
# ---------------------------------------------------------------------------
TOOL_CLASSES: list[str] = [
    "needle_driver",
    "cadiere_forceps",
    "prograsp_forceps",
    "monopolar_curved_scissors",
    "bipolar_forceps",
    "stapler",
    "force_bipolar",
    "vessel_sealer",
    "permanent_cautery_hook_spatula",
    "clip_applier",
    "tip_up_fenestrated_grasper",
    "grasping_retractor",
]

TOOL_DISPLAY: dict[str, str] = {
    "needle_driver": "Needle driver",
    "cadiere_forceps": "Cadiere forceps",
    "prograsp_forceps": "Prograsp forceps",
    "monopolar_curved_scissors": "Monopolar curved scissors",
    "bipolar_forceps": "Bipolar forceps",
    "stapler": "Stapler",
    "force_bipolar": "Force bipolar",
    "vessel_sealer": "Vessel sealer",
    "permanent_cautery_hook_spatula": "Permanent cautery hook/spatula",
    "clip_applier": "Clip applier",
    "tip_up_fenestrated_grasper": "Tip-up fenestrated grasper",
    "grasping_retractor": "Grasping retractor",
}

TASK_CLASSES: list[str] = [
    "suturing",
    "uterine_horn",
    "rectal_artery_vein",
    "suspensory_ligaments",
    "skills_application",
    "range_of_motion",
    "retraction_collision_avoidance",
    "other",
]

TASK_DISPLAY: dict[str, str] = {
    "suturing": "Suturing",
    "uterine_horn": "Uterine horn",
    "rectal_artery_vein": "Rectal artery/vein manipulation",
    "suspensory_ligaments": "Suspensory ligaments",
    "skills_application": "General skills application",
    "range_of_motion": "Range of motion",
    "retraction_collision_avoidance": "Retraction and collision avoidance",
    "other": "Other",
}


@dataclass
class TrainingConfig:
    # --- Dataset ---------------------------------------------------------
    dataset_dir: str = field(
        default_factory=lambda: _env("SURGVU_DIR", str(REPO_ROOT / "data" / "surgvu"))
    )
    frames_dir: str = field(
        default_factory=lambda: _env(
            "SURGVU_FRAMES", str(REPO_ROOT / "data" / "surgvu_frames")
        )
    )
    output_dir: str = field(
        default_factory=lambda: _env("SURGVU_OUT", str(REPO_ROOT / "checkpoints"))
    )

    #: Frames per second to sample when decoding. The source is 60 fps and
    #: neighbouring frames are near-duplicates; 1 fps still yields ~3M frames
    #: from 840 hours, far more than tool presence needs.
    extract_fps: float = field(default_factory=lambda: _env("SURGVU_FPS", 1.0))
    num_workers: int = field(default_factory=lambda: _env("SURGVU_WORKERS", 4))

    # --- Splits ----------------------------------------------------------
    #: Splits are by *case*, never by frame. Frames from one operation are
    #: highly correlated, so a random frame split leaks the validation set
    #: into training and inflates every metric.
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int = 42

    # --- Image ------------------------------------------------------------
    image_size: int = field(default_factory=lambda: _env("SURGVU_IMG", 224))
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

    # --- Classes ----------------------------------------------------------
    tool_classes: list[str] = field(default_factory=lambda: list(TOOL_CLASSES))
    task_classes: list[str] = field(default_factory=lambda: list(TASK_CLASSES))

    # --- Optimisation -----------------------------------------------------
    learning_rate: float = field(default_factory=lambda: _env("SURGVU_LR", 1e-4))
    weight_decay: float = 0.05
    epochs: int = field(default_factory=lambda: _env("SURGVU_EPOCHS", 30))
    batch_size: int = field(default_factory=lambda: _env("SURGVU_BS", 32))
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    early_stopping_patience: int = 7
    amp: bool = True

    # --- Temporal ---------------------------------------------------------
    clip_length: int = field(default_factory=lambda: _env("SURGVU_CLIP", 16))
    #: Stride between sampled frames within a clip, in sampled-frame units.
    frame_step: int = 2
    clip_stride: int = 8  # stride between consecutive training clips

    # --- Inference --------------------------------------------------------
    tool_threshold: float = 0.5

    def tool_display(self, name: str) -> str:
        return TOOL_DISPLAY.get(name, name.replace("_", " ").title())

    def task_display(self, name: str) -> str:
        return TASK_DISPLAY.get(name, name.replace("_", " ").title())

    def as_dict(self) -> dict:
        return {
            "dataset_dir": self.dataset_dir,
            "frames_dir": self.frames_dir,
            "extract_fps": self.extract_fps,
            "image_size": self.image_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "clip_length": self.clip_length,
            "frame_step": self.frame_step,
            "val_fraction": self.val_fraction,
            "test_fraction": self.test_fraction,
            "split_seed": self.split_seed,
            "amp": self.amp,
        }


config = TrainingConfig()
