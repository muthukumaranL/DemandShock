"""Compute Mean Absolute Scaled Error (MASE) for demand forecasts.

MASE is scale-free and compares forecast error with an in-sample naive error,
making it useful when evaluating many item-store series with different volumes.
Values below 1 mean the forecast beats the chosen naive scaling baseline.
"""
from __future__ import annotations

import numpy as np


def mase(actual, predicted, training, seasonality: int = 1) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    training = np.asarray(training, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise ValueError("actual and predicted must be aligned one-dimensional arrays")
    if actual.size == 0:
        raise ValueError("actual and predicted cannot be empty")
    if seasonality < 1 or training.size <= seasonality:
        raise ValueError("training history must be longer than seasonality")
    if not (np.all(np.isfinite(actual)) and np.all(np.isfinite(predicted)) and np.all(np.isfinite(training))):
        raise ValueError("inputs must contain only finite values")

    scale = np.mean(np.abs(training[seasonality:] - training[:-seasonality]))
    if scale == 0:
        raise ValueError("naive scaling error is zero; MASE is undefined")
    return float(np.mean(np.abs(actual - predicted)) / scale)


def summarize_mase(actual, predicted, training, seasonality: int = 1) -> dict:
    score = mase(actual, predicted, training, seasonality)
    return {
        "mase": score,
        "beats_naive_baseline": score < 1.0,
        "seasonality": seasonality,
    }


if __name__ == "__main__":
    history = np.array([18, 20, 19, 22, 21, 23, 22, 25, 24, 26, 25, 28], dtype=float)
    actual = np.array([27, 29, 28, 31], dtype=float)
    predicted = np.array([27.5, 28.0, 29.0, 30.5], dtype=float)
    print(summarize_mase(actual, predicted, history, seasonality=1))
