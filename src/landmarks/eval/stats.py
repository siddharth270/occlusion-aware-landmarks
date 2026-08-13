# Siddharth Mehta, CS5330 PRCV, Final Project
# The significance tests. Both of them compare two models on the same images, which
# makes them more sensitive than treating the two scores as unrelated.

from __future__ import annotations

import math

import numpy as np

from landmarks.eval.gap import global_average_precision


# Bootstrap confidence interval for a single model's GAP.
def bootstrap_gap_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    confs: np.ndarray,
    iters: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    labels, preds, confs = map(np.asarray, (labels, preds, confs))
    n = len(labels)
    point = global_average_precision(labels, preds, confs)

    rng = np.random.default_rng(seed)
    samples = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        samples[i] = global_average_precision(labels[idx], preds[idx], confs[idx])

    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


# Confidence interval and p-value for the GAP difference between two
# models scored on the same images.
def paired_bootstrap_delta(
    labels: np.ndarray,
    preds_a: np.ndarray, confs_a: np.ndarray,
    preds_b: np.ndarray, confs_b: np.ndarray,
    iters: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    labels = np.asarray(labels)
    n = len(labels)

    gap_a = global_average_precision(labels, preds_a, confs_a)
    gap_b = global_average_precision(labels, preds_b, confs_b)
    delta = gap_b - gap_a

    rng = np.random.default_rng(seed)
    deltas = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        deltas[i] = (
            global_average_precision(labels[idx], preds_b[idx], confs_b[idx])
            - global_average_precision(labels[idx], preds_a[idx], confs_a[idx])
        )

    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    opposite = np.mean(deltas <= 0) if delta > 0 else np.mean(deltas >= 0)
    return {
        "gap_a": float(gap_a),
        "gap_b": float(gap_b),
        "delta": float(delta),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(1.0, 2 * opposite)),
        "significant": bool(lo > 0 or hi < 0),
    }


# Two sided exact binomial p-value against a fair coin.
def _binom_two_sided_p(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    total = 1 << n
    lower = sum(math.comb(n, i) for i in range(0, k + 1))
    upper = sum(math.comb(n, i) for i in range(k, n + 1))
    return float(min(1.0, 2.0 * min(lower, upper) / total))


# Upper tail p-value for a chi-square with one degree of freedom.
def _chi2_1df_p(chi2: float) -> float:
    return float(math.erfc(math.sqrt(chi2 / 2.0)))


# McNemar test on which images each model got right, using only the
# ones where the two disagree.
def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    correct_a = np.asarray(correct_a).astype(bool)
    correct_b = np.asarray(correct_b).astype(bool)

    b_only = int(np.sum(~correct_a & correct_b))
    a_only = int(np.sum(correct_a & ~correct_b))
    n_disc = a_only + b_only

    if n_disc == 0:
        return {"a_only": 0, "b_only": 0, "n_discordant": 0,
                "p_value": 1.0, "test": "none", "significant": False}

    if n_disc <= 1000:
        p, test = _binom_two_sided_p(b_only, n_disc), "exact-binomial"
    else:
        chi2 = (abs(b_only - a_only) - 1) ** 2 / n_disc
        p, test = _chi2_1df_p(chi2), "chi2-yates"

    return {"a_only": a_only, "b_only": b_only, "n_discordant": n_disc,
            "p_value": p, "test": test, "significant": bool(p < 0.05)}
