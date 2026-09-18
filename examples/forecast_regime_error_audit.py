"""Slice forecast errors by demand regime to expose hidden failure modes."""

from __future__ import annotations

import numpy as np


def demand_regime_error_audit(
    actual: np.ndarray,
    forecast: np.ndarray,
    *,
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
) -> dict[str, dict[str, float]]:
    """Report MAE and signed bias for low, normal, and high demand periods."""
    y = np.asarray(actual, dtype=float)
    yhat = np.asarray(forecast, dtype=float)
    if y.shape != yhat.shape or y.size == 0:
        raise ValueError("actual and forecast must have the same non-empty shape")
    if not 0 < low_quantile < high_quantile < 1:
        raise ValueError("quantiles must satisfy 0 < low < high < 1")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(yhat)):
        raise ValueError("inputs must contain only finite values")

    low, high = np.quantile(y, [low_quantile, high_quantile])
    masks = {
        "low": y <= low,
        "normal": (y > low) & (y < high),
        "high": y >= high,
    }
    report: dict[str, dict[str, float]] = {}
    for name, mask in masks.items():
        error = yhat[mask] - y[mask]
        report[name] = {
            "count": float(mask.sum()),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
        }
    return report


if __name__ == "__main__":
    actual = np.array([8, 10, 9, 20, 22, 18, 40, 48, 55], dtype=float)
    forecast = np.array([9, 11, 10, 19, 24, 17, 35, 43, 47], dtype=float)
    print(demand_regime_error_audit(actual, forecast))
