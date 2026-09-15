"""Forecast Value Added (FVA) diagnostics.

FVA asks whether each forecasting step actually improves on a simple baseline.
Negative FVA is a useful signal that model complexity is hurting decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def forecast_value_added(actual, baseline, candidates: dict[str, object]) -> pd.DataFrame:
    actual = np.asarray(actual, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if actual.shape != baseline.shape:
        raise ValueError("actual and baseline must have identical shapes")

    baseline_mae = _mae(actual, baseline)
    rows = [{"forecast": "baseline", "mae": baseline_mae, "fva_abs": 0.0, "fva_pct": 0.0}]
    for name, values in candidates.items():
        pred = np.asarray(values, dtype=float)
        if pred.shape != actual.shape:
            raise ValueError(f"{name} has a different shape from actual")
        mae = _mae(actual, pred)
        fva_abs = baseline_mae - mae
        fva_pct = 100 * fva_abs / baseline_mae if baseline_mae else np.nan
        rows.append({"forecast": name, "mae": mae, "fva_abs": fva_abs, "fva_pct": fva_pct})

    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


if __name__ == "__main__":
    actual = np.array([100, 115, 98, 130, 142, 121], dtype=float)
    naive = np.array([98, 100, 115, 98, 130, 142], dtype=float)
    models = {
        "ml_model": [102, 111, 101, 126, 139, 125],
        "manual_override": [105, 120, 108, 139, 150, 130],
    }
    print(forecast_value_added(actual, naive, models).round(2).to_string(index=False))
