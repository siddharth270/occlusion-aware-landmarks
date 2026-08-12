"""Significance testing for the arm comparison.

Two arms trained with one seed each cannot be compared by eyeballing two GAP
numbers: the difference has to be put against the sampling variability of the
test set. Both tests here are *paired* -- the arms are evaluated on identical
images, so pairing removes between-image difficulty as a source of variance and
is far more sensitive than treating the two scores as independent.
"""
from __future__ import annotations

import math

import numpy as np

from landmarks.eval.gap import global_average_precision


def bootstrap_gap_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    confs: np.ndarray,
    iters: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a single arm's GAP.

    Returns (point_estimate, lower, upper).
    """
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


def paired_bootstrap_delta(
    labels: np.ndarray,
    preds_a: np.ndarray, confs_a: np.ndarray,
    preds_b: np.ndarray, confs_b: np.ndarray,
    iters: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """CI and p-value for GAP(b) - GAP(a), resampling images (not predictions).

    The same bootstrap indices are applied to both arms, which is what makes it
    paired. `p_value` is two-sided and estimated as the proportion of resamples
    whose delta has the opposite sign to the observed delta, doubled.
    """
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


def _binom_two_sided_p(k: int, n: int) -> float:
    """Two-sided exact binomial p-value against p=0.5.

    Implemented directly rather than via scipy: the symmetric case reduces to
    2 * min(P(X<=k), P(X>=k)), and keeping this module dependency-free means the
    significance tests cannot fail on a package that isn't installed.
    """
    if n == 0:
        return 1.0
    total = 1 << n                                   # 2**n
    lower = sum(math.comb(n, i) for i in range(0, k + 1))
    upper = sum(math.comb(n, i) for i in range(k, n + 1))
    return float(min(1.0, 2.0 * min(lower, upper) / total))


def _chi2_1df_p(chi2: float) -> float:
    """Upper-tail p for chi-square with 1 df: P(X>x) = erfc(sqrt(x/2))."""
    return float(math.erfc(math.sqrt(chi2 / 2.0)))


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """McNemar test on per-image correctness (top-1, not GAP).

    Only the discordant pairs carry information: images both arms get right or
    both get wrong say nothing about which is better. Reported alongside the
    bootstrap because it tests accuracy rather than ranking, and agreement
    between two different tests is more convincing than either alone.

    Exact binomial for small samples; chi-square with Yates' continuity
    correction once the exact computation would need thousands of big-integer
    binomial coefficients.
    """
    correct_a = np.asarray(correct_a).astype(bool)
    correct_b = np.asarray(correct_b).astype(bool)

    b_only = int(np.sum(~correct_a & correct_b))     # b fixed what a got wrong
    a_only = int(np.sum(correct_a & ~correct_b))     # b broke what a got right
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
