"""Leakage-safe rolling-origin backtesting for time-series forecasting.

A random train/test split is usually invalid for forecasting because it lets
future observations influence model selection. This example builds expanding
or sliding rolling-origin folds so every validation window occurs strictly
after the corresponding training window.

Run:
    python examples/rolling_origin_backtest.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class TimeSeriesFold:
    fold: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def validation_size(self) -> int:
        return self.validation_end - self.validation_start


def rolling_origin_splits(
    n_samples: int,
    *,
    initial_train_size: int,
    horizon: int,
    step: int | None = None,
    expanding: bool = True,
) -> Iterator[TimeSeriesFold]:
    """Yield chronological train/validation index ranges without temporal leakage."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if initial_train_size <= 0:
        raise ValueError("initial_train_size must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    step = horizon if step is None else step
    if step <= 0:
        raise ValueError("step must be positive")
    if initial_train_size + horizon > n_samples:
        raise ValueError("not enough samples for one train/validation fold")

    validation_start = initial_train_size
    fold_number = 1

    while validation_start + horizon <= n_samples:
        train_end = validation_start
        train_start = 0 if expanding else max(0, train_end - initial_train_size)
        validation_end = validation_start + horizon

        fold = TimeSeriesFold(
            fold=fold_number,
            train_start=train_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
        )

        # The key anti-leakage invariant: training ends before validation begins.
        assert fold.train_end <= fold.validation_start
        yield fold

        fold_number += 1
        validation_start += step


def main() -> None:
    # Example: one year of daily observations, first 180 days used for training,
    # then repeated 28-day forecast windows like a retail demand evaluation.
    folds = list(
        rolling_origin_splits(
            365,
            initial_train_size=180,
            horizon=28,
            step=28,
            expanding=True,
        )
    )

    print(f"Generated {len(folds)} leakage-safe folds\n")
    for fold in folds:
        print(
            f"fold={fold.fold} "
            f"train=[{fold.train_start}:{fold.train_end}) "
            f"validation=[{fold.validation_start}:{fold.validation_end}) "
            f"train_n={fold.train_size} validation_n={fold.validation_size}"
        )


if __name__ == "__main__":
    main()
