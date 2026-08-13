# Siddharth Mehta, CS5330 PRCV, Final Project
# Global Average Precision, the competition metric. It ranks predictions across the
# whole test set, so how confident a model is matters as much as what it picks.

from __future__ import annotations

import numpy as np


# Global Average Precision. Every prediction is ranked together, so a
# confident mistake costs more than an uncertain one.
def global_average_precision(
    labels: np.ndarray,
    preds: np.ndarray,
    confs: np.ndarray,
) -> float:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    confs = np.asarray(confs, dtype=np.float64)

    if not (len(labels) == len(preds) == len(confs)):
        raise ValueError("labels, preds and confs must be the same length")
    if len(labels) == 0:
        return 0.0

    order = np.lexsort((np.arange(len(confs)), -confs))
    correct = (preds[order] == labels[order]).astype(np.float64)

    precision_at_rank = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    return float(np.sum(precision_at_rank * correct) / len(labels))


# Plain top-k accuracy, reported next to GAP as a reference that
# ignores confidence.
def top_k_accuracy(labels: np.ndarray, logits: np.ndarray, k: int = 1) -> float:
    labels = np.asarray(labels)
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([labels[i] in topk[i] for i in range(len(labels))]))


# Turns logits into a predicted class plus the softmax probability
# used as its confidence.
def predictions_from_logits(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    return preds, probs[np.arange(len(preds)), preds]
