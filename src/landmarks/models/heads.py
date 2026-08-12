"""Classification heads.

ArcFace is the default because landmark recognition is an instance-level task
with many fine-grained classes: additive angular margin pushes embeddings of the
same landmark together and different landmarks apart far more effectively than
plain softmax, and it is what every strong Google Landmarks solution uses. The
plain Linear head is kept as a control.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """Additive angular margin head (ArcFace, Deng et al. CVPR'19).

    Returns scaled cosine logits. During training the margin is added to the
    target class's angle; at inference (labels=None) it degrades to plain scaled
    cosine similarity, which is what the GAP confidence is derived from.

    Args:
        scale: logit temperature. Too low and the softmax cannot saturate.
        margin: angular margin in radians.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 30.0,
        margin: float = 0.30,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precomputed so the forward pass avoids trig on every batch.
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))

        if labels is None:
            return cosine * self.scale

        cosine = cosine.float()                       # margin math in fp32 under AMP
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Beyond theta = pi - m the margin would decrease the logit, which
        # destabilises training; fall back to a monotonic linear penalty.
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.scale


class LinearHead(nn.Module):
    """Plain softmax classifier, used as the ArcFace control."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None):
        return self.fc(features)
