"""Compare a candidate forecast against simple baselines.

A production forecast should beat naive references, not only report its own
error. This utility calculates MAE/WAPE and relative improvement consistently.
"""
from __future__ import annotations

import numpy as np


def _metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have identical shapes")
    error = np.abs(actual - predicted)
    denom = np.abs(actual).sum()
    return {
        "mae": float(error.mean()),
        "wape": float(error.sum() / denom) if denom else float("nan"),
    }


def compare_forecasts(actual, candidate, **baselines):
    candidate_metrics = _metrics(actual, candidate)
    report = {"candidate": candidate_metrics, "baselines": {}}
    for name, prediction in baselines.items():
        baseline_metrics = _metrics(actual, prediction)
        baseline_mae = baseline_metrics["mae"]
        improvement = (
            (baseline_mae - candidate_metrics["mae"]) / baseline_mae
            if baseline_mae else float("nan")
        )
        report["baselines"][name] = {
            **baseline_metrics,
            "mae_improvement_vs_baseline": float(improvement),
        }
    return report


if __name__ == "__main__":
    actual = np.array([20, 22, 19, 24, 27, 26, 29], dtype=float)
    candidate = np.array([21, 21, 20, 23, 26, 27, 28], dtype=float)
    naive = np.array([20, 20, 22, 19, 24, 27, 26], dtype=float)
    seasonal = np.array([19, 22, 18, 23, 25, 25, 28], dtype=float)
    print(compare_forecasts(actual, candidate, naive=naive, seasonal=seasonal))
