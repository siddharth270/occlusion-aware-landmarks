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
    if name not in timm.list_models():
        candidates = timm.list_models(f"*{name.split('.')[0].split('_')[-1]}*", pretrained=True)
        raise ValueError(
            f"timm has no model {name!r} (timm {timm.__version__}). "
            f"Similar available names: {candidates[:10]}"
        )

    backbone = EmbeddingBackbone(
        name=name,
        pretrained=cfg.model.pretrained,
        embedding_dim=cfg.model.embedding_dim,
        dropout=cfg.model.dropout,
    )

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
