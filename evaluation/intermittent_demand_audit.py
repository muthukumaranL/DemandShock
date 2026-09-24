"""Diagnostics for intermittent-demand forecasts."""
from __future__ import annotations

from collections.abc import Sequence


def audit_intermittent_demand(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    """Measure zero-demand classification and positive-demand magnitude error."""
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted must have equal non-zero length")
    actual_zero = [value == 0 for value in actual]
    predicted_zero = [value <= 0 for value in predicted]
    zero_accuracy = sum(a == p for a, p in zip(actual_zero, predicted_zero)) / len(actual)
    positive = [i for i, value in enumerate(actual) if value > 0]
    mae_positive = (
        sum(abs(float(actual[i]) - float(predicted[i])) for i in positive) / len(positive)
        if positive else 0.0
    )
    false_demand = sum(a and not p for a, p in zip(actual_zero, predicted_zero)) / len(actual)
    missed_demand = sum((not a) and p for a, p in zip(actual_zero, predicted_zero)) / len(actual)
    return {
        "zero_state_accuracy": zero_accuracy,
        "positive_demand_mae": mae_positive,
        "false_demand_rate": false_demand,
        "missed_demand_rate": missed_demand,
        "positive_periods": float(len(positive)),
    }
