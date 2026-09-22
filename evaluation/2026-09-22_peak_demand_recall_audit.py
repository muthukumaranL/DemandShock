"""Evaluate whether a demand forecast identifies operationally important peaks."""

from __future__ import annotations

import numpy as np


def peak_demand_recall(
    actual: np.ndarray,
    predicted: np.ndarray,
    peak_quantile: float = 0.9,
) -> dict[str, float | int]:
    """Measure recall/precision for high-demand periods and their magnitude bias."""
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or y.size == 0:
        raise ValueError("actual and predicted must be equal non-empty 1D arrays")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        raise ValueError("inputs must be finite")
    if not 0.5 < peak_quantile < 1.0:
        raise ValueError("peak_quantile must be between 0.5 and 1.0")

    actual_cutoff = float(np.quantile(y, peak_quantile))
    predicted_cutoff = float(np.quantile(p, peak_quantile))
    actual_peak = y >= actual_cutoff
    predicted_peak = p >= predicted_cutoff
    tp = int(np.sum(actual_peak & predicted_peak))
    actual_count = int(np.sum(actual_peak))
    predicted_count = int(np.sum(predicted_peak))
    recall = tp / actual_count if actual_count else 0.0
    precision = tp / predicted_count if predicted_count else 0.0
    peak_bias = float(np.mean(p[actual_peak] - y[actual_peak])) if actual_count else 0.0

    return {
        "actual_peak_threshold": actual_cutoff,
        "predicted_peak_threshold": predicted_cutoff,
        "actual_peak_periods": actual_count,
        "predicted_peak_periods": predicted_count,
        "matched_peak_periods": tp,
        "peak_recall": float(recall),
        "peak_precision": float(precision),
        "peak_magnitude_bias": peak_bias,
    }


if __name__ == "__main__":
    actual = np.array([10, 11, 9, 14, 12, 30, 13, 15, 35, 12, 40, 11], dtype=float)
    forecast = np.array([11, 10, 10, 13, 14, 27, 14, 16, 25, 13, 38, 12], dtype=float)
    print(peak_demand_recall(actual, forecast, peak_quantile=0.8))