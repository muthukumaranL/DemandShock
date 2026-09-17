"""Summarize rolling-origin forecast performance and stability."""

from __future__ import annotations

import numpy as np


def rolling_origin_audit(
    actual_by_fold: list[np.ndarray],
    forecast_by_fold: list[np.ndarray],
) -> dict[str, float | list[float]]:
    """Report fold-level MAE/WAPE and how much accuracy varies over time."""
    if len(actual_by_fold) != len(forecast_by_fold) or not actual_by_fold:
        raise ValueError("actual and forecast fold lists must be non-empty and aligned")

    maes: list[float] = []
    wapes: list[float] = []
    for actual, forecast in zip(actual_by_fold, forecast_by_fold):
        y = np.asarray(actual, dtype=float)
        yhat = np.asarray(forecast, dtype=float)
        if y.shape != yhat.shape or y.size == 0:
            raise ValueError("each actual/forecast fold must have the same non-empty shape")
        error = np.abs(y - yhat)
        maes.append(float(error.mean()))
        denominator = float(np.abs(y).sum())
        wapes.append(float(error.sum() / denominator) if denominator > 0 else np.nan)

    valid_wape = np.asarray(wapes, dtype=float)
    return {
        "fold_mae": maes,
        "fold_wape": wapes,
        "mean_mae": float(np.mean(maes)),
        "mae_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
        "mean_wape": float(np.nanmean(valid_wape)),
        "worst_fold_mae": float(np.max(maes)),
    }


if __name__ == "__main__":
    actual = [np.array([10, 12, 11]), np.array([14, 13, 16]), np.array([18, 17, 20])]
    forecast = [np.array([11, 11, 12]), np.array([13, 14, 15]), np.array([16, 18, 18])]
    print(rolling_origin_audit(actual, forecast))
