from __future__ import annotations

import numpy as np


def expected_calibration_error(pred_probs: np.ndarray, outcomes: np.ndarray, n_buckets: int = 10) -> float:
    pred_probs = np.asarray(pred_probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if pred_probs.shape != outcomes.shape:
        raise ValueError("pred_probs and outcomes must have same shape")

    bins = np.linspace(0, 1, n_buckets + 1)
    total = len(pred_probs)
    ece = 0.0
    for i in range(n_buckets):
        lo, hi = bins[i], bins[i + 1]
        mask = (pred_probs >= lo) & (pred_probs < hi if i < n_buckets - 1 else pred_probs <= hi)
        if not np.any(mask):
            continue
        bucket_prob = pred_probs[mask].mean()
        bucket_outcome = outcomes[mask].mean()
        ece += (mask.sum() / total) * abs(bucket_prob - bucket_outcome)
    return float(ece)


def mae(pred_means: np.ndarray, actuals: np.ndarray) -> float:
    pred_means = np.asarray(pred_means, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    return float(np.mean(np.abs(pred_means - actuals)))


def centering_rate(pred_means: np.ndarray, actuals: np.ndarray) -> float:
    """If line equals predicted mean, returns over hit-rate."""
    pred_means = np.asarray(pred_means, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    return float(np.mean(actuals > pred_means))
