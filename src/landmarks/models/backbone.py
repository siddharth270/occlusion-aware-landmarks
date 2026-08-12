"""timm feature extractor plus embedding neck."""
from __future__ import annotations

import torch
import torch.nn as nn


class EmbeddingBackbone(nn.Module):
    """Pretrained CNN reduced to a fixed-size embedding.

    The neck (BN -> dropout -> linear -> BN) is the standard recipe for metric
    learning heads: the trailing BatchNorm keeps embedding magnitudes stable,
    which matters because ArcFace L2-normalises them.
    """

    def __init__(
        self,
        name: str = "efficientnet_b0",
        pretrained: bool = True,
        embedding_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        import timm

        # num_classes=0 + global_pool="avg" gives pooled features, no classifier.
        self.trunk = timm.create_model(
            name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        trunk_dim = self.trunk.num_features

        self.neck = nn.Sequential(
            nn.BatchNorm1d(trunk_dim),
            nn.Dropout(dropout),
            nn.Linear(trunk_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neck(self.trunk(x))
