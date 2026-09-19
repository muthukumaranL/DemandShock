"""Measure how forecast quality degrades as the prediction horizon grows."""

from __future__ import annotations

import numpy as np


def horizon_error_audit(actual: np.ndarray, forecast: np.ndarray) -> list[dict[str, float]]:
    """Report MAE, RMSE, bias, and WAPE for every forecast horizon."""
    y = np.asarray(actual, dtype=float)
    yhat = np.asarray(forecast, dtype=float)
    if y.shape != yhat.shape or y.ndim != 2 or y.size == 0:
        raise ValueError("actual and forecast must be equal non-empty 2D arrays")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(yhat)):
        raise ValueError("inputs must contain only finite values")

    rows: list[dict[str, float]] = []
    for step in range(y.shape[1]):
        error = yhat[:, step] - y[:, step]
        denom = float(np.sum(np.abs(y[:, step])))
        rows.append({
            "horizon": float(step + 1),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "bias": float(np.mean(error)),
            "wape": float(np.sum(np.abs(error)) / denom) if denom else float("nan"),
        })
    return rows


def degradation_ratio(rows: list[dict[str, float]]) -> float:
    """Compare final-horizon MAE with first-horizon MAE."""
    if not rows or rows[0]["mae"] == 0:
        return float("nan")
    return float(rows[-1]["mae"] / rows[0]["mae"])


if __name__ == "__main__":
    actual = np.array([[20, 22, 25, 28], [12, 14, 16, 19], [40, 43, 45, 48]], dtype=float)
    forecast = np.array([[21, 23, 27, 32], [12, 15, 18, 23], [39, 44, 49, 55]], dtype=float)
    report = horizon_error_audit(actual, forecast)
    print(report)
    print("final/first MAE:", degradation_ratio(report))
