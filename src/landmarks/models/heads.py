# Siddharth Mehta, CS5330 PRCV, Final Project
# The classification heads. ArcFace is the one used, because the task has a thousand
# very similar classes and it separates them better than a plain classifier.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):

    # Sets up the class weights and precomputes the trig terms the
    # margin needs.
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

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    # Returns scaled cosine logits, with the angular margin added to the
    # true class during training.
    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))

        if labels is None:
            return cosine * self.scale

        cosine = cosine.float()
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.scale


class LinearHead(nn.Module):

    # Sets up the class weights and precomputes the trig terms the
    # margin needs.
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    # Returns scaled cosine logits, with the angular margin added to the
    # true class during training.
    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None):
        return self.fc(features)
