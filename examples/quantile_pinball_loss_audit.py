"""Evaluate probabilistic demand forecasts with quantile pinball loss."""

from __future__ import annotations

import numpy as np


def quantile_loss_audit(
    actual: np.ndarray,
    forecasts: np.ndarray,
    quantiles: list[float],
) -> list[dict[str, float]]:
    """Report pinball loss and directional miss rates for each forecast quantile."""
    y = np.asarray(actual, dtype=float)
    pred = np.asarray(forecasts, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    if y.ndim != 1 or y.size == 0 or pred.shape != (y.size, q.size):
        raise ValueError("forecasts must have shape observations x quantiles")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(pred)):
        raise ValueError("actual and forecasts must be finite")
    if q.size == 0 or np.any((q <= 0) | (q >= 1)) or len(set(q.tolist())) != q.size:
        raise ValueError("quantiles must be unique values strictly between 0 and 1")

    rows: list[dict[str, float]] = []
    for idx, quantile in enumerate(q):
        error = y - pred[:, idx]
        loss = np.maximum(quantile * error, (quantile - 1.0) * error)
        rows.append({
            "quantile": float(quantile),
            "mean_pinball_loss": float(np.mean(loss)),
            "underforecast_rate": float(np.mean(error > 0)),
            "overforecast_rate": float(np.mean(error < 0)),
            "empirical_below_quantile": float(np.mean(y <= pred[:, idx])),
        })
    return rows


if __name__ == "__main__":
    actual = np.array([12, 18, 15, 24, 20], dtype=float)
    predictions = np.array([[10, 12, 15], [14, 18, 22], [12, 16, 20], [18, 23, 28], [16, 20, 25]], dtype=float)
    print(quantile_loss_audit(actual, predictions, [0.1, 0.5, 0.9]))
