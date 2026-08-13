# Siddharth Mehta, CS5330 PRCV, Final Project
# The pretrained image encoder and the small stack of layers that turns its output
# into a fixed size embedding.

from __future__ import annotations

import torch
import torch.nn as nn


class EmbeddingBackbone(nn.Module):

    # Builds the pretrained trunk and the layers that reduce it to a
    # fixed size embedding.
    def __init__(
        self,
        name: str = "efficientnet_b0",
        pretrained: bool = True,
        embedding_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        import timm

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

    # Turns a batch of images into embeddings.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neck(self.trunk(x))
