"""Evaluate empirical coverage of forecast intervals.

Point-error metrics do not tell whether uncertainty bands are trustworthy. This
example computes interval coverage and average width using synthetic forecasts.
It is standalone and does not modify DemandShock artifacts.

Run:
    python examples/forecast_interval_coverage.py
"""

from __future__ import annotations

import numpy as np


def interval_metrics(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[float, float]:
    if not (len(y_true) == len(lower) == len(upper)):
        raise ValueError("all inputs must have equal length")
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")

    covered = (y_true >= lower) & (y_true <= upper)
    coverage = float(np.mean(covered))
    average_width = float(np.mean(upper - lower))
    return coverage, average_width


def main() -> None:
    rng = np.random.default_rng(42)
    actual = rng.normal(loc=100.0, scale=12.0, size=1000)
    prediction = actual + rng.normal(loc=0.0, scale=7.0, size=1000)

    half_width = 1.645 * 7.0  # illustrative nominal 90% interval
    lower = prediction - half_width
    upper = prediction + half_width

    coverage, width = interval_metrics(actual, lower, upper)
    print(f"Empirical coverage: {coverage:.1%}")
    print(f"Average interval width: {width:.2f}")
    print("Target nominal coverage: 90.0%")


if __name__ == "__main__":
    main()
