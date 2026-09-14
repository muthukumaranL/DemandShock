"""Audit whether bottom-level forecasts reconcile to their aggregate forecast.

Retail forecasts are often generated independently at item, store, category,
and total levels. Independent models can disagree. This utility measures the
coherence gap before a reconciliation step is applied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def hierarchy_coherence_audit(
    bottom_forecasts: Mapping[str, Sequence[float]],
    aggregate_forecast: Sequence[float],
) -> dict[str, float]:
    if not bottom_forecasts:
        raise ValueError("bottom_forecasts cannot be empty")

    aggregate = [float(value) for value in aggregate_forecast]
    if not aggregate:
        raise ValueError("aggregate_forecast cannot be empty")

    series = {name: [float(value) for value in values] for name, values in bottom_forecasts.items()}
    if any(len(values) != len(aggregate) for values in series.values()):
        raise ValueError("all forecast series must share the same horizon")

    reconstructed = [sum(values[i] for values in series.values()) for i in range(len(aggregate))]
    gaps = [reconstructed[i] - aggregate[i] for i in range(len(aggregate))]
    abs_gaps = [abs(gap) for gap in gaps]
    aggregate_scale = sum(abs(value) for value in aggregate)

    return {
        "horizon": float(len(aggregate)),
        "bottom_series": float(len(series)),
        "mean_absolute_coherence_gap": sum(abs_gaps) / len(abs_gaps),
        "max_absolute_coherence_gap": max(abs_gaps),
        "signed_coherence_bias": sum(gaps) / len(gaps),
        "relative_coherence_gap": sum(abs_gaps) / aggregate_scale if aggregate_scale else 0.0,
        "fully_coherent_period_share": sum(gap < 1e-9 for gap in abs_gaps) / len(abs_gaps),
    }


if __name__ == "__main__":
    bottom = {
        "store_a": [105, 110, 118, 121],
        "store_b": [72, 75, 79, 83],
        "store_c": [48, 51, 50, 54],
    }
    total_model = [230, 239, 251, 265]

    print("Hierarchical forecast coherence audit")
    for metric, value in hierarchy_coherence_audit(bottom, total_model).items():
        print(f"{metric:>32}: {value:.4f}")
