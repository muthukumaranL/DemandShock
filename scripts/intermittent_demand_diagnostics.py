"""Characterize intermittent demand before choosing a forecasting strategy.

Reports zero-demand share, average inter-demand interval (ADI), squared
coefficient of variation (CV²) for non-zero demand, and the common
smooth/intermittent/erratic/lumpy demand classification.
"""
from __future__ import annotations

import numpy as np


def demand_diagnostics(values) -> dict:
    demand = np.asarray(values, dtype=float)
    if demand.ndim != 1 or demand.size < 2:
        raise ValueError("values must be a one-dimensional series with at least two points")
    if not np.all(np.isfinite(demand)) or np.any(demand < 0):
        raise ValueError("demand must contain finite non-negative values")

    positive_idx = np.flatnonzero(demand > 0)
    positive = demand[positive_idx]
    zero_share = float(np.mean(demand == 0))
    if positive.size == 0:
        return {
            "zero_share": 1.0,
            "nonzero_periods": 0,
            "adi": float("inf"),
            "cv2": float("inf"),
            "demand_type": "all_zero",
        }

    # ADI includes the implied interval over the observed horizon.
    adi = float(demand.size / positive.size)
    mean_positive = float(np.mean(positive))
    cv2 = float((np.std(positive, ddof=1) / mean_positive) ** 2) if positive.size > 1 else 0.0

    intermittent = adi >= 1.32
    variable = cv2 >= 0.49
    if intermittent and variable:
        demand_type = "lumpy"
    elif intermittent:
        demand_type = "intermittent"
    elif variable:
        demand_type = "erratic"
    else:
        demand_type = "smooth"

    return {
        "zero_share": zero_share,
        "nonzero_periods": int(positive.size),
        "adi": adi,
        "cv2": cv2,
        "demand_type": demand_type,
    }


if __name__ == "__main__":
    example = [0, 0, 4, 0, 0, 0, 7, 0, 2, 0, 0, 0, 9, 0, 0, 3]
    print(demand_diagnostics(example))
