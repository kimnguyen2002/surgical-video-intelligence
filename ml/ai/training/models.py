"""
SurgVU model architectures.

**Tool detection** — a timm backbone (ConvNeXt-V2, EfficientNetV2, ViT, Swin,
DINOv2) with a multi-label head. Sigmoid per class, not softmax: several
instruments are installed simultaneously, so the classes are not mutually
exclusive.

**Step recognition** — a temporal model over a clip. Three backends, in
descending order of capability and ascending order of how likely they are to
actually run on the machine in front of you:

``timesformer`` / ``videomae``
    True video transformers via HuggingFace ``transformers``. Best accuracy,
    heaviest requirements.
``cnn_gru`` *(default)*
    A shared 2-D backbone extracts per-frame features, and a bidirectional GRU
    plus attention pooling models the temporal structure. Trains on a single
    consumer GPU, and because the frame encoder is an ordinary timm model it
    can be initialised from the tool-detection checkpoint — the surgical
    features transfer directly.
``mean_pool``
    The same frame encoder with mean pooling and no temporal model. Its only
    purpose is as an ablation baseline: if a temporal model cannot beat it,
    the temporal modelling is not earning its complexity.

Every model exposes ``gradcam_layer()`` so explainability works uniformly.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from ai.training.config import config

logger = logging.getLogger("surgvu.models")


# ---------------------------------------------------------------------------
# Tool detection
# ---------------------------------------------------------------------------
class SurgVUToolModel(nn.Module):
    """Multi-label instrument presence classifier."""

    def __init__(
        self,
        model_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        num_classes: int = len(config.tool_classes),
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        import timm

        self.model_name = model_name
        self.num_classes = num_classes

        try:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        except Exception as exc:
            # Usually: no network on first run, or an unknown model name.
            logger.warning(
                "Could not create '%s' with pretrained=%s (%s); retrying without "
                "pretrained weights.",
                model_name,
                pretrained,
                exc,
            )
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)

        self.embed_dim = int(
            getattr(self.backbone, "num_features", None) or self.backbone.embed_dim
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W) → (B, num_classes)`` logits."""
        return self.head(self.backbone(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled embeddings — reused by the temporal memory index."""
        return self.backbone(x)

    def gradcam_layer(self):
        from ai.explainability import find_target_layer

        return find_target_layer(self, self.model_name)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: Optional[float] = None) -> dict:
        threshold = config.tool_threshold if threshold is None else threshold
        probabilities = torch.sigmoid(self(x))
        return {
            "probabilities": probabilities,
            "present": (probabilities >= threshold),
        }


# ---------------------------------------------------------------------------
# Step recognition
# ---------------------------------------------------------------------------
class TemporalAttentionPool(nn.Module):
    """
    Learned attention pooling over time.

    Mean pooling dilutes short but decisive moments — a two-second clip
    application inside a sixteen-frame window. Attention lets the model weight
    the frames that actually determine the task.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)  # (B, T)
        pooled = torch.einsum("bt,btd->bd", weights, x)
        return pooled, weights


class SurgVUStepModel(nn.Module):
    """
    Temporal surgical task classifier.

    Input ``(B, T, C, H, W)``; output ``(B, num_classes)`` logits.
    """

    def __init__(
        self,
        model_name: str = "cnn_gru",
        num_classes: int = len(config.task_classes),
        frame_encoder: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        clip_length: int = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.clip_length = clip_length or config.clip_length
        self.frame_encoder_name = frame_encoder
        self.kind = "video_transformer" if model_name in {"timesformer", "videomae"} else model_name

        if self.kind == "video_transformer":
            self._build_video_transformer(model_name, num_classes)
        else:
            self._build_frame_encoder(frame_encoder, pretrained)
            if model_name == "mean_pool":
                self.temporal = None
                feature_dim = self.embed_dim
            else:  # cnn_gru
                self.temporal = nn.GRU(
                    input_size=self.embed_dim,
                    hidden_size=hidden_dim,
                    num_layers=2,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout,
                )
                feature_dim = hidden_dim * 2
                self.pool = TemporalAttentionPool(feature_dim)
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )

    def _build_frame_encoder(self, name: str, pretrained: bool) -> None:
        import timm

        try:
            self.encoder = timm.create_model(name, pretrained=pretrained, num_classes=0)
        except Exception as exc:
            logger.warning("Falling back to untrained '%s' (%s)", name, exc)
            self.encoder = timm.create_model(name, pretrained=False, num_classes=0)
        self.embed_dim = int(
            getattr(self.encoder, "num_features", None) or self.encoder.embed_dim
        )

    def _build_video_transformer(self, name: str, num_classes: int) -> None:
        checkpoints = {
            "timesformer": "facebook/timesformer-base-finetuned-k400",
            "videomae": "MCG-NJU/videomae-base-finetuned-kinetics",
        }
        try:
            from transformers import AutoModelForVideoClassification

            self.video_model = AutoModelForVideoClassification.from_pretrained(
                checkpoints[name],
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        except Exception as exc:
            raise ImportError(
                f"'{name}' needs the transformers library and its pretrained "
                f"checkpoint ({exc}).\n"
                "  pip install transformers\n"
                "Or train the dependency-free temporal model instead:\n"
                "  python -m ai.training.train_step_recognition --model cnn_gru"
            ) from exc

    def load_frame_encoder_from_tool_model(self, checkpoint_path: str) -> bool:
        """
        Warm-start the frame encoder from a trained tool detector.

        Tool detection learns exactly the features step recognition needs —
        instruments, tissue, and how they interact — from far more supervision
        than the task labels provide. This usually converges faster and higher
        than starting from ImageNet.
        """
        if self.kind == "video_transformer":
            logger.warning("Video transformers cannot load a 2-D tool encoder.")
            return False
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            weights = state.get("model", state)
            encoder_weights = {
                key[len("backbone.") :]: value
                for key, value in weights.items()
                if key.startswith("backbone.")
            }
            if not encoder_weights:
                logger.warning("No 'backbone.*' weights in %s", checkpoint_path)
                return False
            missing, unexpected = self.encoder.load_state_dict(encoder_weights, strict=False)
            logger.info(
                "Warm-started frame encoder from %s (%d missing, %d unexpected)",
                checkpoint_path,
                len(missing),
                len(unexpected),
            )
            return True
        except Exception as exc:
            logger.error("Could not warm-start from %s: %s", checkpoint_path, exc)
            return False

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if self.kind == "video_transformer":
            # HuggingFace video models expect (B, T, C, H, W).
            return self.video_model(pixel_values=x).logits

        batch, frames = x.shape[0], x.shape[1]
        # Fold time into the batch so the 2-D encoder sees a flat batch.
        flat = x.flatten(0, 1)
        features = self.encoder(flat).view(batch, frames, self.embed_dim)

        if self.temporal is None:
            pooled = features.mean(dim=1)
            attention = None
        else:
            sequence, _ = self.temporal(features)
            pooled, attention = self.pool(sequence)

        logits = self.head(pooled)
        if return_attention:
            return logits, attention
        return logits

    def gradcam_layer(self):
        """Grad-CAM targets the frame encoder — spatial evidence per frame."""
        from ai.explainability import find_target_layer

        if self.kind == "video_transformer":
            return None
        return find_target_layer(
            type("Wrapper", (), {"backbone": self.encoder})(), self.frame_encoder_name
        )


# ---------------------------------------------------------------------------
# Factory / checkpoint helpers
# ---------------------------------------------------------------------------
def build_model(task: str, **kwargs) -> nn.Module:
    if task == "tool":
        return SurgVUToolModel(**kwargs)
    if task == "step":
        return SurgVUStepModel(**kwargs)
    raise ValueError(f"Unknown task '{task}'. Expected 'tool' or 'step'.")


def save_checkpoint(path, model: nn.Module, meta: dict) -> None:
    """
    Save weights together with everything needed to rebuild the model.

    A bare ``state_dict`` is not enough: loading it requires knowing the
    architecture, class list, and image size. Storing them here means
    ``load_checkpoint`` can reconstruct the model from the file alone.
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_name": getattr(model, "model_name", ""),
            "num_classes": getattr(model, "num_classes", 0),
            "image_size": config.image_size,
            **meta,
        },
        path,
    )


def load_checkpoint(path, task: str = "tool", device: str = "cpu", **overrides):
    """Rebuild a model from a checkpoint written by :func:`save_checkpoint`."""
    state = torch.load(path, map_location=device, weights_only=False)

    kwargs = {"pretrained": False}
    if state.get("model_name"):
        kwargs["model_name"] = state["model_name"]
    if state.get("num_classes"):
        kwargs["num_classes"] = state["num_classes"]
    if task == "step" and state.get("frame_encoder"):
        kwargs["frame_encoder"] = state["frame_encoder"]
    kwargs.update(overrides)

    model = build_model(task, **kwargs)

    # Shipped checkpoints store weights in float16 to keep the repository
    # small (see scripts/compress_models.py). Widen them explicitly rather
    # than relying on load_state_dict's implicit copy_ cast: inference runs in
    # float32 on CPU, and a model left holding half-precision parameters
    # raises "expected scalar type Half but found Float" on the first forward
    # pass — at request time, not at load time, which is the worst place for it.
    weights = {
        key: (value.float() if getattr(value, "dtype", None) == torch.float16 else value)
        for key, value in state["model"].items()
    }

    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        logger.warning(
            "Checkpoint mismatch: %d missing, %d unexpected keys", len(missing), len(unexpected)
        )
    model.to(device).eval()
    return model, state
