"""Audit whether a demand forecast captures promotional uplift.

Aggregate accuracy can hide a model that systematically under-forecasts
promotional periods. This utility reports bias and uplift capture separately
for promotion and non-promotion observations.
"""

from __future__ import annotations

import numpy as np


def promotion_uplift_audit(
    actual: np.ndarray,
    forecast: np.ndarray,
    is_promo: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    is_promo = np.asarray(is_promo, dtype=bool)
    if not (actual.shape == forecast.shape == is_promo.shape):
        raise ValueError("actual, forecast and is_promo must have matching shapes")
    if actual.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    if not is_promo.any() or is_promo.all():
        raise ValueError("both promo and non-promo observations are required")

    promo_actual = float(actual[is_promo].mean())
    base_actual = float(actual[~is_promo].mean())
    promo_forecast = float(forecast[is_promo].mean())
    base_forecast = float(forecast[~is_promo].mean())

    actual_uplift = promo_actual - base_actual
    forecast_uplift = promo_forecast - base_forecast
    capture = forecast_uplift / actual_uplift if abs(actual_uplift) > 1e-12 else np.nan

    return {
        "promo_bias": float((forecast[is_promo] - actual[is_promo]).mean()),
        "non_promo_bias": float((forecast[~is_promo] - actual[~is_promo]).mean()),
        "actual_promo_uplift": actual_uplift,
        "forecast_promo_uplift": forecast_uplift,
        "uplift_capture_ratio": float(capture),
    }


if __name__ == "__main__":
    y = np.array([10, 11, 10, 18, 20, 9, 10, 17, 19, 11], dtype=float)
    yhat = np.array([10, 10, 11, 15, 16, 10, 10, 15, 16, 11], dtype=float)
    promo = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
    print(promotion_uplift_audit(y, yhat, promo))
