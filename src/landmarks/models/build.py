"""Assemble backbone + head into the recognition model."""
from __future__ import annotations

import torch
import torch.nn as nn

from landmarks.models.backbone import EmbeddingBackbone
from landmarks.models.heads import ArcMarginProduct, LinearHead


class LandmarkNet(nn.Module):
    """Embedding backbone with a margin or linear classification head.

    `forward` takes labels because ArcFace needs the target class to apply its
    margin during training. At eval time labels are omitted and the head returns
    plain scaled cosine logits.
    """

    def __init__(self, backbone: EmbeddingBackbone, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.backbone(x), labels)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalised embeddings, for retrieval-style analysis in the report."""
        return nn.functional.normalize(self.backbone(x))


def build_model(cfg, num_classes: int) -> LandmarkNet:
    """Construct the model described by `cfg.model`.

    Raises a diagnostic error if the timm name is unavailable, since model names
    were renamed in timm 0.9 and a silent fallback would make the two arms
    incomparable.
    """
    import timm

    name = cfg.model.backbone

    # Try to build rather than pre-checking against list_models(): that returns
    # architecture names WITHOUT pretrained tags, so a valid "arch.weights"
    # string never matches and the check rejects working models.
    try:
        backbone = EmbeddingBackbone(
            name=name,
            pretrained=cfg.model.pretrained,
            embedding_dim=cfg.model.embedding_dim,
            dropout=cfg.model.dropout,
        )
    except Exception as exc:
        arch = name.split(".")[0]
        candidates = timm.list_models(f"*{arch}*", pretrained=True)
        raise ValueError(
            f"could not build timm model {name!r} (timm {timm.__version__}): {exc}. "
            f"Pretrained names matching {arch!r}: {candidates[:10]}"
        ) from exc

    if cfg.model.head == "arcface":
        head: nn.Module = ArcMarginProduct(
            in_features=cfg.model.embedding_dim,
            out_features=num_classes,
            scale=cfg.model.arcface.scale,
            margin=cfg.model.arcface.margin,
        )
    elif cfg.model.head == "linear":
        head = LinearHead(cfg.model.embedding_dim, num_classes)
    else:
        raise ValueError(f"unknown head {cfg.model.head!r}; expected arcface|linear")

    return LandmarkNet(backbone, head)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
