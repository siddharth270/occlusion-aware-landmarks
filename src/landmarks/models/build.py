# Siddharth Mehta, CS5330 PRCV, Final Project
# Puts the backbone and the head together into the model that every arm trains.

from __future__ import annotations

import torch
import torch.nn as nn

from landmarks.models.backbone import EmbeddingBackbone
from landmarks.models.heads import ArcMarginProduct, LinearHead


class LandmarkNet(nn.Module):

    # Holds the backbone and the classification head together.
    def __init__(self, backbone: EmbeddingBackbone, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    # Returns class logits. Labels are needed during training because
    # ArcFace adds its margin to the true class.
    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.backbone(x), labels)

    # Returns normalised embeddings, for inspecting the representation directly.
    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.backbone(x))


# Assembles the backbone and head described by the config.
def build_model(cfg, num_classes: int) -> LandmarkNet:
    import timm

    name = cfg.model.backbone

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


# Counts the trainable parameters.
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
