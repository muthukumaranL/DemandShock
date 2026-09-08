"""Audit forecast error by prediction horizon.

Run with demo data:
    python examples/forecast_horizon_bias.py

Run on a CSV:
    python examples/forecast_horizon_bias.py --csv predictions.csv

Expected columns: horizon, y_true, y_pred. Reports MAE, signed bias and WAPE
for each forecast horizon so degradation at longer lead times is visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"horizon", "y_true", "y_pred"}


def horizon_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    data = frame[list(REQUIRED_COLUMNS)].dropna().copy()
    if data.empty:
        raise ValueError("no complete prediction rows to evaluate")

    data["abs_error"] = (data["y_pred"] - data["y_true"]).abs()
    data["signed_error"] = data["y_pred"] - data["y_true"]

    rows: list[dict[str, float | int]] = []
    for horizon, group in data.groupby("horizon", sort=True):
        denominator = group["y_true"].abs().sum()
        wape = np.nan if denominator == 0 else group["abs_error"].sum() / denominator
        rows.append(
            {
                "horizon": int(horizon),
                "n": int(len(group)),
                "mae": float(group["abs_error"].mean()),
                "bias": float(group["signed_error"].mean()),
                "wape": float(wape),
            }
        )
    return pd.DataFrame(rows)


def demo_predictions(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for horizon in range(1, 8):
        actual = rng.gamma(shape=4.0, scale=8.0, size=80)
        noise = rng.normal(loc=0.35 * horizon, scale=2.0 + 0.55 * horizon, size=80)
        predicted = np.clip(actual + noise, 0.0, None)
        rows.extend(
            {"horizon": horizon, "y_true": y, "y_pred": p}
            for y, p in zip(actual, predicted, strict=True)
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="optional prediction CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.csv) if args.csv else demo_predictions()
    report = horizon_metrics(frame)
    print(report.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    worst = report.loc[report["mae"].idxmax()]
    print(
        f"\nHighest MAE occurs at horizon {int(worst['horizon'])}: "
        f"MAE={worst['mae']:.4f}, bias={worst['bias']:.4f}"
    )


if __name__ == "__main__":
    main()
