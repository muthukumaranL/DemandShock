"""Forecasting models: two naive benchmarks and one global LightGBM.

Strategy: a single DIRECT multi-horizon model. Every feature is frozen at the
forecast origin `O = t - 28`, so one vectorised predict call covers the whole
28-day path and horizons 7/14/28 are prefixes of it. No recursive feedback loop
exists, which is deliberate:

  * with ~50-64% zero-demand days, feeding fractional predictions back into
    integer-heavy lag features compounds distribution shift over 28 steps and
    would contaminate the residuals the shock module depends on;
  * training and inference share ONE feature function, so there is no second
    implementation to drift out of sync.

The honest cost, stated in the app: a 7-day-ahead forecast sees demand only as of
28 days earlier. Seasonal-naive-28 shares exactly that information set, which is
why it is the primary benchmark.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import Config
from .features import categorical_columns, feature_columns

MODEL_LABELS = {
    "naive": "Naive (last observed day)",
    "snaive7": "Seasonal Naive (lag 7)",
    "snaive28": "Seasonal Naive (lag 28)",
    "lgbm": "LightGBM (global, Tweedie)",
}
BASELINE_MODELS = ["naive", "snaive7", "snaive28"]


@dataclass
class FoldSpec:
    name: str
    train_end_d: int
    val_start_d: int
    val_end_d: int

    @property
    def horizon(self) -> int:
        return self.val_end_d - self.val_start_d + 1

    @property
    def has_actuals(self) -> bool:
        return self.name != "FORWARD"


def fold_specs(cfg: Config, names: list[str] | None = None) -> list[FoldSpec]:
    chosen = names or list(cfg.folds)
    return [FoldSpec(name=n, **cfg.fold(n)) for n in chosen]


@dataclass
class TrainedModel:
    booster: lgb.Booster
    features: list[str]
    categorical: list[str]
    config_name: str
    fold: str
    best_iteration: int
    train_rows: int
    train_seconds: float
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def baseline_forecast(sales: pd.DataFrame, fold: FoldSpec, model: str) -> pd.DataFrame:
    """Naive benchmarks computed from actuals at or before the forecast origin.

      naive     : f(O+h) = y(O)                      for every h
      snaive7   : f(O+h) = y(O + h - 7*ceil(h/7))    - last 7 real days, tiled
      snaive28  : f(O+h) = y(O + h - 28)             - same information set as LightGBM
    """
    origin = fold.train_end_d
    horizon = fold.horizon
    steps = np.arange(1, horizon + 1)
    if model == "naive":
        offsets = np.zeros(horizon, dtype=int)
    elif model == "snaive7":
        offsets = steps - 7 * np.ceil(steps / 7).astype(int)
    elif model == "snaive28":
        offsets = steps - 28
    else:
        raise ValueError(f"Unknown baseline model {model!r}")

    # Select only the handful of source days the benchmark repeats. Every offset is
    # non-positive, so these are all at or before the origin by construction - there
    # is no need to materialise the whole pre-origin history first.
    needed = sorted({origin + int(o) for o in offsets})
    source = sales[sales["d"].isin(needed)]
    if source.empty:
        return pd.DataFrame(columns=["item_id", "store_id", "d", "step", "y_pred"])
    wide = source.pivot_table(index=["item_id", "store_id"], columns="d",
                              values="sales", observed=True)
    # A series may lack a needed day only if it was released after it; those days
    # carry no demand history, so the benchmark has nothing to repeat.
    wide = wide.reindex(columns=needed)

    keys = wide.index.to_frame(index=False)
    item_keys = keys["item_id"].astype(str).to_numpy()
    store_keys = keys["store_id"].astype(str).to_numpy()
    parts = []
    for step, offset in zip(steps, offsets):
        column = origin + int(offset)
        parts.append(pd.DataFrame({
            "item_id": item_keys,
            "store_id": store_keys,
            "d": origin + int(step),
            "step": int(step),
            "y_pred": wide[column].to_numpy(dtype="float64"),
        }))
    out = pd.concat(parts, ignore_index=True)
    out["y_pred"] = out["y_pred"].clip(lower=0)
    return out.dropna(subset=["y_pred"])


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------
def _slice_training_rows(frame: pd.DataFrame, fold: FoldSpec,
                         window: int | None) -> pd.DataFrame:
    train = frame[frame["d"] <= fold.train_end_d]
    if window:
        train = train[train["d"] > fold.train_end_d - window]
    return train.dropna(subset=["sales"])


def train_lgbm(cfg: Config, frame: pd.DataFrame, fold: FoldSpec, config_name: str,
               params_override: dict[str, Any] | None = None,
               num_boost_round: int | None = None,
               use_early_stopping: bool = True,
               logger=None) -> TrainedModel:
    """Fit the global model for one fold and one ablation arm.

    Early stopping uses the chronologically LAST `early_stopping_days` days of the
    training span - never a random split, and never the fold's validation window.
    """
    cols = feature_columns(config_name)
    cats = categorical_columns(config_name)
    params = cfg.lgbm_params()
    if params_override:
        params.update(params_override)
    # A Tweedie-only parameter left behind by a caller would be meaningless (and
    # confusing in the recorded metadata) under any other objective.
    if params.get("objective") != "tweedie":
        params.pop("tweedie_variance_power", None)

    train = _slice_training_rows(frame, fold, cfg.train_window_days)
    if train.empty:
        raise ValueError(f"No training rows for fold {fold.name}")

    started = time.time()
    if use_early_stopping:
        cutoff = fold.train_end_d - cfg.early_stopping_days
        fit_mask = train["d"] <= cutoff
        es_mask = train["d"] > cutoff
        if fit_mask.sum() == 0 or es_mask.sum() == 0:
            use_early_stopping = False

    rounds = num_boost_round or cfg.num_boost_round
    if use_early_stopping:
        dtrain = lgb.Dataset(train.loc[fit_mask, cols], label=train.loc[fit_mask, "sales"],
                             categorical_feature=cats, free_raw_data=True)
        dvalid = lgb.Dataset(train.loc[es_mask, cols], label=train.loc[es_mask, "sales"],
                             categorical_feature=cats, reference=dtrain, free_raw_data=True)
        booster = lgb.train(
            params, dtrain, num_boost_round=rounds, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
        )
        best = int(booster.best_iteration or rounds)
    else:
        dtrain = lgb.Dataset(train[cols], label=train["sales"],
                             categorical_feature=cats, free_raw_data=True)
        booster = lgb.train(params, dtrain, num_boost_round=rounds)
        best = int(rounds)

    elapsed = time.time() - started
    if logger:
        logger.info(
            "    fit %s/%s: %s rows, %s features, best_iter=%s, %.1fs",
            config_name, fold.name, f"{len(train):,}", len(cols), best, elapsed)
    return TrainedModel(
        booster=booster, features=cols, categorical=cats, config_name=config_name,
        fold=fold.name, best_iteration=best, train_rows=int(len(train)),
        train_seconds=float(elapsed), params=params,
    )


def predict_window(model: TrainedModel, frame: pd.DataFrame, fold: FoldSpec,
                   min_step: int | None = None,
                   max_step: int | None = None) -> pd.DataFrame:
    """Score the fold's window in one vectorised call.

    `min_step`/`max_step` restrict scoring to a horizon bucket: a model trained
    with shift S may only serve steps up to S, because at step S its newest input
    is exactly the forecast origin.
    """
    start = fold.val_start_d + (min_step - 1 if min_step else 0)
    end = fold.val_start_d + (max_step - 1) if max_step else fold.val_end_d
    window = frame[(frame["d"] >= max(start, fold.val_start_d))
                   & (frame["d"] <= min(end, fold.val_end_d))]
    if window.empty:
        return pd.DataFrame(columns=["item_id", "store_id", "d", "step",
                                     "y_true", "y_pred"])
    preds = model.booster.predict(
        window[model.features], num_iteration=model.best_iteration)
    return pd.DataFrame({
        "item_id": window["item_id"].astype(str).to_numpy(),
        "store_id": window["store_id"].astype(str).to_numpy(),
        "d": window["d"].to_numpy(),
        "step": (window["d"] - fold.train_end_d).to_numpy().astype("int16"),
        "y_true": window["sales"].to_numpy(dtype="float64"),
        "y_pred": np.clip(preds, 0, None),
    })


def fit_predict_buckets(cfg: Config, fold: FoldSpec, config_name: str,
                        load_frame, params_override: dict[str, Any] | None = None,
                        num_boost_round: int | None = None,
                        use_early_stopping: bool = True,
                        logger=None) -> tuple[pd.DataFrame, list[TrainedModel]]:
    """Train one model per horizon bucket and stitch the 28-day path back together.

    Each bucket uses the freshest demand history its own last step allows: a model
    serving steps 1-7 may read up to the forecast origin, so its features are
    shifted 7 days rather than 28. `load_frame(shift)` supplies that bucket's
    feature table.

    Returns the concatenated predictions (one row per item-store-step, exactly as
    a single-model run produced) plus the fitted models.
    """
    parts, models = [], []
    for bucket in cfg.horizon_buckets:
        shift = int(bucket["shift"])
        frame = load_frame(shift)
        model = train_lgbm(cfg, frame, fold, config_name,
                           params_override=params_override,
                           num_boost_round=num_boost_round,
                           use_early_stopping=use_early_stopping, logger=logger)
        preds = predict_window(model, frame, fold,
                               min_step=int(bucket["min_step"]),
                               max_step=int(bucket["max_step"]))
        if not preds.empty:
            parts.append(preds)
        models.append(model)
        if logger:
            logger.info("      bucket %s (steps %s-%s, shift %s): %s rows scored",
                        bucket["name"], bucket["min_step"], bucket["max_step"],
                        shift, f"{len(preds):,}")
        del frame
    if not parts:
        return pd.DataFrame(columns=["item_id", "store_id", "d", "step",
                                     "y_true", "y_pred"]), models
    return pd.concat(parts, ignore_index=True), models


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------
def fit_residual_quantiles(cfg: Config, residuals: pd.DataFrame,
                           hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Empirical residual quantiles by (store, category, predicted-level bucket).

    Chosen over three quantile-objective boosters because those triple the training
    budget to mostly learn "zero" on intermittent series, whereas out-of-sample
    residuals are calibrated by construction. Sparse groups fall back to
    (category, bucket) and then to the global bucket, so every group is backed by
    at least `min_group_residuals` real observations.
    """
    edges = list(cfg["uncertainty"]["prediction_buckets"]) + [np.inf]
    quantiles = list(cfg["uncertainty"]["quantiles"])
    min_n = int(cfg["uncertainty"]["min_group_residuals"])

    df = residuals.merge(hierarchy[["item_id", "store_id", "cat_id"]],
                         on=["item_id", "store_id"], how="left")
    df["residual"] = df["y_true"] - df["y_pred"]
    df["bucket"] = pd.cut(df["y_pred"], bins=edges, right=False,
                          labels=range(len(edges) - 1)).astype("Int16")

    def quantile_table(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        grouped = frame.groupby(keys, observed=True)["residual"]
        out = grouped.quantile(quantiles).unstack()
        out.columns = [f"q{int(q * 100):02d}" for q in quantiles]
        out["n"] = grouped.size()
        return out.reset_index()

    global_tbl = quantile_table(df, ["bucket"]).rename(
        columns={c: f"{c}_global" for c in ("q10", "q50", "q90", "n")})
    cat_tbl = quantile_table(df, ["cat_id", "bucket"]).rename(
        columns={c: f"{c}_cat" for c in ("q10", "q50", "q90", "n")})
    full_tbl = quantile_table(df, ["store_id", "cat_id", "bucket"])

    table = full_tbl.merge(cat_tbl, on=["cat_id", "bucket"], how="left")
    table = table.merge(global_tbl, on="bucket", how="left")

    for q in ("q10", "q50", "q90"):
        chosen = np.where(table["n"] >= min_n, table[q],
                          np.where(table["n_cat"] >= min_n, table[f"{q}_cat"],
                                   table[f"{q}_global"]))
        table[q] = chosen.astype("float64")
    table["source"] = np.where(
        table["n"] >= min_n, "store_cat_bucket",
        np.where(table["n_cat"] >= min_n, "cat_bucket", "global_bucket"))

    keep = ["store_id", "cat_id", "bucket", "q10", "q50", "q90", "n", "source"]
    out = table[keep].copy()
    # Monotonicity: a P10 above P90 would be nonsense to display.
    out[["q10", "q50", "q90"]] = np.sort(
        out[["q10", "q50", "q90"]].to_numpy(dtype="float64"), axis=1)
    return out


def apply_residual_quantiles(cfg: Config, forecasts: pd.DataFrame,
                             quantiles: pd.DataFrame,
                             hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Attach P10/P90 (and a calibrated P50) to a forecast frame."""
    edges = list(cfg["uncertainty"]["prediction_buckets"]) + [np.inf]
    out = forecasts.merge(hierarchy[["item_id", "store_id", "cat_id"]],
                          on=["item_id", "store_id"], how="left")
    out["bucket"] = pd.cut(out["y_pred"], bins=edges, right=False,
                           labels=range(len(edges) - 1)).astype("Int16")
    out = out.merge(quantiles[["store_id", "cat_id", "bucket", "q10", "q50", "q90"]],
                    on=["store_id", "cat_id", "bucket"], how="left")
    for q in ("q10", "q50", "q90"):
        out[q] = out[q].fillna(0.0)
    out["p10"] = np.clip(out["y_pred"] + out["q10"], 0, None)
    out["p50"] = np.clip(out["y_pred"] + out["q50"], 0, None)
    out["p90"] = np.clip(out["y_pred"] + out["q90"], 0, None)
    out[["p10", "p50", "p90"]] = np.sort(
        out[["p10", "p50", "p90"]].to_numpy(dtype="float64"), axis=1)
    return out.drop(columns=["q10", "q50", "q90", "bucket", "cat_id"])
