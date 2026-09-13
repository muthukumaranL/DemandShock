"""Evaluate forecasts with asymmetric stockout and overstock costs.

Forecast error metrics treat over- and under-forecasting similarly, but retail
operations usually do not. This utility converts forecast errors into a simple
business cost so model selection can reflect service-level priorities.
"""

from __future__ import annotations

import numpy as np


def asymmetric_inventory_cost(
    actual: np.ndarray,
    forecast: np.ndarray,
    *,
    stockout_cost_per_unit: float = 3.0,
    overstock_cost_per_unit: float = 1.0,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if actual.shape != forecast.shape:
        raise ValueError("actual and forecast must have the same shape")
    if actual.size == 0:
        raise ValueError("actual and forecast cannot be empty")
    if stockout_cost_per_unit < 0 or overstock_cost_per_unit < 0:
        raise ValueError("costs must be non-negative")

    under = np.maximum(actual - forecast, 0.0)
    over = np.maximum(forecast - actual, 0.0)
    stockout_cost = under.sum() * stockout_cost_per_unit
    overstock_cost = over.sum() * overstock_cost_per_unit
    total_cost = stockout_cost + overstock_cost

    demand = float(actual.sum())
    fulfilled = float(np.minimum(actual, forecast).sum())
    service_level = fulfilled / demand if demand > 0 else 1.0

    return {
        "total_cost": float(total_cost),
        "stockout_cost": float(stockout_cost),
        "overstock_cost": float(overstock_cost),
        "underforecast_units": float(under.sum()),
        "overforecast_units": float(over.sum()),
        "service_level": float(service_level),
        "cost_per_period": float(total_cost / actual.size),
    }


if __name__ == "__main__":
    actual = np.array([20, 25, 18, 40, 35, 22, 30], dtype=float)
    candidate_a = np.array([19, 22, 20, 34, 31, 24, 28], dtype=float)
    candidate_b = np.array([23, 28, 21, 42, 37, 25, 33], dtype=float)

    for name, forecast in {"lean_forecast": candidate_a, "buffered_forecast": candidate_b}.items():
        result = asymmetric_inventory_cost(actual, forecast)
        print(f"\n{name}")
        for metric, value in result.items():
            print(f"{metric:>22}: {value:.3f}")
