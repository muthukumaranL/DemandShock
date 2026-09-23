"""Translate demand forecasts into inventory service-level diagnostics."""

from __future__ import annotations

import numpy as np


def inventory_service_level_audit(
    actual: np.ndarray,
    forecast: np.ndarray,
    safety_stock: float = 0.0,
) -> dict[str, float | int]:
    """Measure stockout frequency, fill rate, excess units, and inventory bias."""
    demand = np.asarray(actual, dtype=float)
    pred = np.asarray(forecast, dtype=float)
    if demand.ndim != 1 or pred.ndim != 1 or demand.shape != pred.shape or demand.size == 0:
        raise ValueError("actual and forecast must be equal non-empty 1D arrays")
    if not np.all(np.isfinite(demand)) or not np.all(np.isfinite(pred)):
        raise ValueError("inputs must be finite")
    if np.any(demand < 0) or np.any(pred < 0) or safety_stock < 0:
        raise ValueError("demand, forecast, and safety_stock must be non-negative")

    available = pred + float(safety_stock)
    shortage = np.maximum(demand - available, 0.0)
    excess = np.maximum(available - demand, 0.0)
    stockout = shortage > 0
    total_demand = float(np.sum(demand))
    units_filled = float(np.sum(demand - shortage))

    return {
        "periods": int(demand.size),
        "stockout_periods": int(np.sum(stockout)),
        "cycle_service_level": float(1.0 - np.mean(stockout)),
        "fill_rate": float(units_filled / total_demand) if total_demand > 0 else 1.0,
        "shortage_units": float(np.sum(shortage)),
        "excess_units": float(np.sum(excess)),
        "mean_inventory_bias": float(np.mean(available - demand)),
        "safety_stock": float(safety_stock),
    }


if __name__ == "__main__":
    demand = np.array([12, 15, 19, 11, 28, 17, 31, 14], dtype=float)
    forecast = np.array([13, 14, 17, 13, 24, 19, 27, 15], dtype=float)
    print(inventory_service_level_audit(demand, forecast, safety_stock=2.0))
