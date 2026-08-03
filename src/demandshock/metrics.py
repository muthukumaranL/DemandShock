"""Forecast accuracy metrics.

Exact definitions, degenerate cases handled explicitly and visibly. Nothing here
silently drops a series: every exclusion is counted and surfaced in the app.

A note on naming: this project does NOT implement the official M5 WRMSSE (the
12-level, 42,840-series weighted aggregate). The name is therefore never used.
What is reported is per-series RMSSE and clearly-labelled aggregates of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_NAMES = ["mae", "rmse", "rmsse", "wape", "smape", "bias"]

METRIC_HELP = {
    "mae": "Mean absolute error, in units, averaged over every item-day.",
    "rmse": "Root mean squared error, in units. Penalises large misses harder.",
    "rmsse": ("Root mean squared scaled error. Per series, the forecast's squared "
              "error divided by the squared error of a one-day naive forecast over "
              "the training period. Below 1 beats that naive benchmark."),
    "wape": ("Weighted absolute percentage error: total absolute error divided by "
             "total actual units. Robust on intermittent demand."),
    "smape": ("Symmetric mean absolute percentage error. Days where actual and "
              "forecast are both zero count as perfect (0)."),
    "bias": ("Total forecast minus total actual, as a share of total actual. "
             "Positive means over-forecasting."),
}


# ---------------------------------------------------------------------------
# scaling denominator for RMSSE
# ---------------------------------------------------------------------------
def naive_squared_diffs(sales: pd.DataFrame) -> pd.DataFrame:
    """Squared one-step-naive errors per series, sorted once.

    Split out from `rmsse_scales` so a multi-fold backtest sorts and differences
    the full sales history a single time instead of once per fold.
    """
    ordered = sales.sort_values(["store_id", "item_id", "d"], kind="stable")
    diffs = ordered.groupby(["store_id", "item_id"], observed=True)["sales"].diff()
    return ordered.assign(sq=diffs.to_numpy() ** 2)[
        ["item_id", "store_id", "d", "sq"]]


def scales_from_diffs(diffs: pd.DataFrame, train_end_d: int) -> pd.DataFrame:
    """Mean squared naive error up to `train_end_d`, per series."""
    train = diffs[diffs["d"] <= train_end_d]
    scales = (
        train.groupby(["item_id", "store_id"], observed=True)["sq"]
        .mean().reset_index().rename(columns={"sq": "scale"})
    )
    scales["scale"] = scales["scale"].astype("float64")
    for col in ("item_id", "store_id"):
        if isinstance(scales[col].dtype, pd.CategoricalDtype):
            scales[col] = scales[col].astype(str)
    return scales


def rmsse_scales(sales: pd.DataFrame, train_end_d: int) -> pd.DataFrame:
    """Per-series mean squared one-step naive error over the training period.

    Uses only post-release days at or before `train_end_d` (the melt already
    dropped pre-release rows). Series whose training history is flat - including
    all-zero series - have a scale of 0; RMSSE is undefined for them and they are
    excluded from RMSSE aggregates with the exclusion count reported.
    """
    return scales_from_diffs(naive_squared_diffs(sales), train_end_d)


# ---------------------------------------------------------------------------
# elementwise pieces
# ---------------------------------------------------------------------------
def _smape_terms(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    denom = np.abs(y) + np.abs(f)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(denom == 0, 0.0, 200.0 * np.abs(y - f) / denom)
    return terms


def pooled_metrics(y: np.ndarray, f: np.ndarray) -> dict[str, float]:
    """Scale-pooled metrics over a set of item-days (RMSSE handled separately)."""
    y = np.asarray(y, dtype="float64")
    f = np.asarray(f, dtype="float64")
    err = f - y
    abs_err = np.abs(err)
    total_actual = y.sum()
    return {
        "mae": float(abs_err.mean()) if len(y) else float("nan"),
        "rmse": float(np.sqrt((err ** 2).mean())) if len(y) else float("nan"),
        "wape": float(abs_err.sum() / total_actual) if total_actual > 0 else float("nan"),
        "smape": float(_smape_terms(y, f).mean()) if len(y) else float("nan"),
        "bias": float(err.sum() / total_actual) if total_actual > 0 else float("nan"),
    }


def series_rmsse(frame: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    """Per-series RMSSE. NaN where the training history was flat (scale == 0)."""
    per = (
        frame.assign(sq=(frame["y_pred"] - frame["y_true"]) ** 2)
        .groupby(["item_id", "store_id"], observed=True)["sq"].mean()
        .reset_index().rename(columns={"sq": "mse"})
    )
    per = per.merge(scales, on=["item_id", "store_id"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        per["rmsse"] = np.where(
            (per["scale"] > 0) & np.isfinite(per["scale"]),
            np.sqrt(per["mse"] / per["scale"]),
            np.nan,
        )
    return per[["item_id", "store_id", "rmsse", "scale"]]


# ---------------------------------------------------------------------------
# the main entry point
# ---------------------------------------------------------------------------
def compute_metrics(frame: pd.DataFrame, scales: pd.DataFrame,
                    group_cols: list[str] | None = None) -> pd.DataFrame:
    """Metrics for a forecast frame, optionally split by hierarchy columns.

    `frame` needs: item_id, store_id, y_true, y_pred (+ any grouping columns).
    Aggregation rules:
      * mae / rmse / smape : pooled over every item-day in the group
      * wape / bias        : ratio of summed absolute error (or error) to summed actual
      * rmsse              : unweighted mean of the per-series values in the group
    """
    frame = frame.dropna(subset=["y_true", "y_pred"])
    if frame.empty:
        return pd.DataFrame(columns=(group_cols or []) + METRIC_NAMES
                            + ["n_series", "n_days", "n_obs", "n_rmsse_excluded"])

    per_series = series_rmsse(frame, scales)
    rmsse_lookup = per_series.set_index(["item_id", "store_id"])["rmsse"]

    if not group_cols:
        groups = [((), frame)]
    else:
        groups = list(frame.groupby(group_cols, observed=True, dropna=False))

    rows = []
    for key, block in groups:
        stats = pooled_metrics(block["y_true"].to_numpy(), block["y_pred"].to_numpy())
        keys = block[["item_id", "store_id"]].drop_duplicates()
        values = rmsse_lookup.reindex(
            pd.MultiIndex.from_frame(keys)).to_numpy(dtype="float64")
        finite = values[np.isfinite(values)]
        stats["rmsse"] = float(finite.mean()) if finite.size else float("nan")
        stats["n_series"] = int(len(keys))
        stats["n_rmsse_excluded"] = int(len(values) - finite.size)
        stats["n_days"] = int(block["d"].nunique()) if "d" in block.columns else 0
        stats["n_obs"] = int(len(block))
        if group_cols:
            key_tuple = key if isinstance(key, tuple) else (key,)
            stats.update(dict(zip(group_cols, key_tuple)))
        rows.append(stats)

    out = pd.DataFrame(rows)
    ordered = (group_cols or []) + METRIC_NAMES + [
        "n_series", "n_days", "n_obs", "n_rmsse_excluded"]
    return out[[c for c in ordered if c in out.columns]]


def metrics_by_horizon(frame: pd.DataFrame, scales: pd.DataFrame,
                       horizons: list[int], group_cols: list[str] | None = None
                       ) -> pd.DataFrame:
    """Metrics restricted to the first h forecast steps, for each horizon h."""
    parts = []
    for horizon in horizons:
        subset = frame[frame["step"] <= horizon]
        block = compute_metrics(subset, scales, group_cols)
        if block.empty:
            continue
        block.insert(0, "horizon", horizon)
        parts.append(block)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def forecast_accuracy(wape: float) -> float:
    """Business-facing accuracy score. Clamped at 0: WAPE can exceed 100% on
    intermittent series, and a negative 'accuracy' would be meaningless."""
    if wape is None or not np.isfinite(wape):
        return float("nan")
    return float(max(0.0, 100.0 * (1.0 - wape)))


def interval_coverage(frame: pd.DataFrame, lo: str = "p10", hi: str = "p90") -> dict:
    """Empirical coverage of a prediction interval, measured on actuals."""
    sub = frame.dropna(subset=["y_true", lo, hi])
    if sub.empty:
        return {"coverage": float("nan"), "below": float("nan"),
                "above": float("nan"), "n": 0}
    inside = ((sub["y_true"] >= sub[lo]) & (sub["y_true"] <= sub[hi])).mean()
    return {
        "coverage": float(inside),
        "below": float((sub["y_true"] < sub[lo]).mean()),
        "above": float((sub["y_true"] > sub[hi]).mean()),
        "n": int(len(sub)),
    }
