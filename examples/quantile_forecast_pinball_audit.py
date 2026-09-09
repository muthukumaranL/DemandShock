"""Evaluate probabilistic forecasts with pinball loss and interval sharpness.

Run:
    python examples/quantile_forecast_pinball_audit.py

The demo scores p10/p50/p90 forecasts, reports pinball loss for each quantile,
and checks empirical 80% interval coverage and width. Lower pinball loss is
better; coverage should be interpreted together with interval sharpness.
"""
from __future__ import annotations

import numpy as np


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    error = y_true - y_pred
    return float(np.mean(np.maximum(q * error, (q - 1.0) * error)))


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[float, float]:
    covered = (y_true >= lower) & (y_true <= upper)
    return float(covered.mean()), float(np.mean(upper - lower))


def demo() -> None:
    rng = np.random.default_rng(42)
    actual = rng.poisson(lam=8.0, size=400).astype(float)
    median = np.clip(actual + rng.normal(0, 2.0, size=actual.size), 0, None)
    p10 = np.clip(median - 3.0, 0, None)
    p90 = median + 3.0

    for q, forecast in ((0.1, p10), (0.5, median), (0.9, p90)):
        print(f"q={q:.1f} pinball_loss={pinball_loss(actual, forecast, q):.4f}")
    coverage, width = interval_metrics(actual, p10, p90)
    print(f"80% interval coverage={coverage:.3f}")
    print(f"average interval width={width:.3f}")


if __name__ == "__main__":
    demo()
