"""Global Average Precision -- the Google Landmarks competition metric.

    GAP = (1/M) * sum_i P(i) * rel(i)

where predictions across ALL images are sorted by confidence descending, P(i) is
precision at rank i, rel(i) is 1 if prediction i is correct, and M is the number
of images that contain a landmark.

The crucial property is that GAP is global, not per-image: a model that is
confidently wrong is punished harder than one that is uncertainly wrong, because
the confident error pollutes precision for every lower-ranked prediction. That
makes confidence calibration part of the metric, which is exactly what should
change when occluders are removed from the input.

In this closed-set setup every evaluation image has a label, so M == len(labels).
"""
from __future__ import annotations

import numpy as np


def global_average_precision(
    labels: np.ndarray,
    preds: np.ndarray,
    confs: np.ndarray,
) -> float:
    """GAP over one set of single-prediction-per-image results.

    Args:
        labels: ground-truth class per image, shape (N,).
        preds: predicted class per image, shape (N,).
        confs: confidence of each prediction, shape (N,).
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    confs = np.asarray(confs, dtype=np.float64)

    if not (len(labels) == len(preds) == len(confs)):
        raise ValueError("labels, preds and confs must be the same length")
    if len(labels) == 0:
        return 0.0

    # Descending confidence. Ties broken by index for determinism.
    order = np.lexsort((np.arange(len(confs)), -confs))
    correct = (preds[order] == labels[order]).astype(np.float64)

    precision_at_rank = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    return float(np.sum(precision_at_rank * correct) / len(labels))


def top_k_accuracy(labels: np.ndarray, logits: np.ndarray, k: int = 1) -> float:
    """Plain accuracy, reported alongside GAP as a confidence-free reference."""
    labels = np.asarray(labels)
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([labels[i] in topk[i] for i in range(len(labels))]))


def predictions_from_logits(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Argmax class and its softmax probability, used as the GAP confidence.

    Softmax rather than the raw cosine logit: GAP ranks predictions across the
    whole set, so the confidence must be comparable between images, and an
    unnormalised margin logit is not.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)   # numerical stability
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    return preds, probs[np.arange(len(preds)), preds]
