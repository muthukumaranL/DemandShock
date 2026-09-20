"""Evaluate forecast interval reliability separately at each prediction horizon."""

from __future__ import annotations

import numpy as np


def interval_horizon_audit(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_coverage: float = 0.90,
) -> list[dict[str, float]]:
    """Report empirical coverage, coverage gap, and interval width by horizon."""
    y = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if y.ndim != 2 or y.shape != lo.shape or y.shape != hi.shape or y.size == 0:
        raise ValueError("actual, lower, and upper must be equal non-empty 2D arrays")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("inputs must contain only finite values")
    if np.any(lo > hi):
        raise ValueError("lower bounds cannot exceed upper bounds")
    if not 0 < nominal_coverage < 1:
        raise ValueError("nominal_coverage must be between 0 and 1")

    rows: list[dict[str, float]] = []
    for step in range(y.shape[1]):
        covered = (y[:, step] >= lo[:, step]) & (y[:, step] <= hi[:, step])
        empirical = float(np.mean(covered))
        rows.append({
            "horizon": float(step + 1),
            "empirical_coverage": empirical,
            "coverage_gap": empirical - nominal_coverage,
            "mean_interval_width": float(np.mean(hi[:, step] - lo[:, step])),
            "miss_rate": float(1.0 - empirical),
        })
    return rows


if __name__ == "__main__":
    actual = np.array([[10, 12, 15], [20, 23, 25], [8, 9, 11]], dtype=float)
    lower = actual - np.array([1.0, 1.5, 2.0])
    upper = actual + np.array([1.0, 1.5, 2.0])
    upper[0, 2] = 14.0
    print(interval_horizon_audit(actual, lower, upper))