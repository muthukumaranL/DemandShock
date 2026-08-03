"""Feature engineering - the canonical feature specification.

THE ONE INVARIANT THAT MATTERS
------------------------------
Every feature for target date `t` is computable from information available at the
forecast origin `O = t - 28 days`, or is genuinely known in advance (retail
calendar, SNAP schedule, published weekly sell price).

That is why:
  * demand lags start at 28 and rolling windows are applied to `y.shift(28)`;
  * FEMA / FRED tables are joined at `date - 28 days` (the state of the world as
    known at the origin), never at the target date;
  * no feature is derived from the target day's own sales.

The consequence, stated honestly in the app: a 7-day-ahead forecast sees demand
only as of 28 days earlier. The gain is that one model serves every horizon, and
residuals mean exactly one thing - "actual vs what was expectable at the origin".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .data import read_processed, read_sales_long

# ---------------------------------------------------------------------------
# canonical feature groups
# ---------------------------------------------------------------------------
DEMAND_FEATURES = [
    "lag_28", "lag_35", "lag_42", "lag_49", "lag_56",
    "roll_mean_7", "roll_mean_28", "roll_mean_56",
    "roll_std_7", "roll_std_28", "roll_std_56",
    "zero_rate_28", "days_since_last_sale", "days_since_release",
]
CALENDAR_FEATURES = [
    "wday", "day_of_month", "week_of_year", "month", "snap", "is_event",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    "days_to_next_event", "days_since_last_event",
]
PRICE_FEATURES = [
    "sell_price", "price_change_pct", "price_rel_52wk",
    "price_momentum_4wk", "price_changed_flag",
]
FEMA_FEATURES = [
    "fema_active_count", "fema_dr_active", "fema_fire_active",
    "fema_flood_active", "fema_storm_active", "fema_counties_active",
    "fema_days_since_declaration",
]
FRED_FEATURES = ["ur_level", "ur_change_1m", "ur_change_3m", "ur_yoy"]
STATIC_FEATURES = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "demand": DEMAND_FEATURES,
    "calendar": CALENDAR_FEATURES,
    "price": PRICE_FEATURES,
    "fema": FEMA_FEATURES,
    "fred": FRED_FEATURES,
    "static": STATIC_FEATURES,
}

# Ablation arms: strictly cumulative, so any accuracy delta is attributable.
ABLATION_SETS: dict[str, list[str]] = {
    "A": ["demand", "static"],
    "B": ["demand", "static", "calendar", "price"],
    "C": ["demand", "static", "calendar", "price", "fema"],
    "D": ["demand", "static", "calendar", "price", "fema", "fred"],
}
ABLATION_LABELS = {
    "A": "Demand history only",
    "B": "+ Calendar & Price",
    "C": "+ FEMA disaster context",
    "D": "+ FRED economic context",
}

CATEGORICAL_FEATURES = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
]

KEY_COLUMNS = ["item_id", "store_id", "d", "date", "sales"]

FEATURE_FAMILY = {}
for _family, _cols in FEATURE_GROUPS.items():
    for _c in _cols:
        FEATURE_FAMILY[_c] = _family

FAMILY_LABELS = {
    "demand": "Demand history",
    "calendar": "Calendar & Events",
    "price": "Price",
    "fema": "FEMA context",
    "fred": "Economic context",
    "static": "Product / Store identity",
}

_NO_EVENT = "__none__"


def feature_columns(config_name: str) -> list[str]:
    """Ordered feature list for an ablation arm ('A'..'D')."""
    try:
        groups = ABLATION_SETS[config_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown feature config {config_name!r}; expected one of "
            f"{sorted(ABLATION_SETS)}"
        ) from exc
    cols: list[str] = []
    for group in groups:
        cols.extend(FEATURE_GROUPS[group])
    return cols


def categorical_columns(config_name: str) -> list[str]:
    cols = set(feature_columns(config_name))
    return [c for c in CATEGORICAL_FEATURES if c in cols]


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _demand_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Lags and rolling statistics, all shifted by `demand_shift` days first.

    Rows are contiguous daily per series (the melt keeps every day from release
    onward), so a row-wise `shift(k)` is exactly a k-day lag.
    """
    spec = cfg["features"]
    shift = int(spec["demand_shift"])
    grp = df.groupby("item_id", observed=True, sort=False)["sales"]

    for lag in spec["lags"]:
        df[f"lag_{lag}"] = grp.shift(lag).astype("float32")

    # Shift BEFORE rolling: every window ends at t - shift, never includes t.
    shifted = grp.shift(shift)
    roll_grp = shifted.groupby(df["item_id"], observed=True, sort=False)
    for window in spec["rolling_windows"]:
        roller = roll_grp.rolling(window, min_periods=max(2, window // 2))
        df[f"roll_mean_{window}"] = (
            roller.mean().reset_index(level=0, drop=True).astype("float32"))
        df[f"roll_std_{window}"] = (
            roller.std().reset_index(level=0, drop=True).astype("float32"))

    zw = int(spec["zero_rate_window"])
    is_zero = (df["sales"] == 0).astype("float32").where(df["sales"].notna())
    zero_shift = is_zero.groupby(df["item_id"], observed=True, sort=False).shift(shift)
    df["zero_rate_28"] = (
        zero_shift.groupby(df["item_id"], observed=True, sort=False)
        .rolling(zw, min_periods=max(2, zw // 2)).mean()
        .reset_index(level=0, drop=True).astype("float32"))

    # Days since the most recent non-zero sale, evaluated at t - shift.
    sold = df["sales"].fillna(0) > 0
    last_sale_d = df["d"].where(sold)
    last_sale_d = last_sale_d.groupby(df["item_id"], observed=True, sort=False).ffill()
    last_sale_shifted = last_sale_d.groupby(
        df["item_id"], observed=True, sort=False).shift(shift)
    origin_d = df["d"] - shift
    cap = float(spec["days_since_last_sale_cap"])
    df["days_since_last_sale"] = (
        (origin_d - last_sale_shifted).astype("float32").fillna(cap).clip(0, cap))

    df["days_since_release"] = (
        (df["d"] - df["release_d"]).astype("float32")
        .clip(0, float(spec["days_since_release_cap"])))
    return df


def _calendar_features(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    cal = calendar.copy()
    cal["day_of_month"] = cal["date"].dt.day.astype("int8")
    cal["week_of_year"] = cal["date"].dt.isocalendar().week.astype("int8")
    cal["is_event"] = (
        cal["event_name_1"].notna() | cal["event_name_2"].notna()).astype("int8")

    # Distance to the nearest event day in each direction (deterministic calendar
    # information, known arbitrarily far ahead).
    event_d = cal.loc[cal["event_name_1"].notna(), "d"].to_numpy()
    all_d = cal["d"].to_numpy()
    idx_next = np.searchsorted(event_d, all_d, side="left")
    idx_prev = np.searchsorted(event_d, all_d, side="right") - 1
    next_d = np.where(idx_next < len(event_d), event_d[np.clip(idx_next, 0, len(event_d) - 1)], np.nan)
    prev_d = np.where(idx_prev >= 0, event_d[np.clip(idx_prev, 0, len(event_d) - 1)], np.nan)
    cal["days_to_next_event"] = np.clip(next_d - all_d, 0, 60).astype("float32")
    cal["days_since_last_event"] = np.clip(all_d - prev_d, 0, 60).astype("float32")
    cal["days_to_next_event"] = cal["days_to_next_event"].fillna(60).astype("float32")
    cal["days_since_last_event"] = cal["days_since_last_event"].fillna(60).astype("float32")

    for col in ("event_name_1", "event_type_1", "event_name_2", "event_type_2"):
        cal[col] = cal[col].fillna(_NO_EVENT).astype(str)

    keep = ["d", "date", "wm_yr_wk", "wday", "month", "day_of_month", "week_of_year",
            "is_event", "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "days_to_next_event", "days_since_last_event",
            "snap_CA", "snap_TX", "snap_WI"]
    before = len(df)
    df = df.merge(cal[keep], on="d", how="left", validate="m:1")
    assert len(df) == before, "calendar join changed row count"

    # SNAP applies to the store's OWN state only.
    state = df["state_id"].astype(str).to_numpy()
    df["snap"] = np.select(
        [state == "CA", state == "TX", state == "WI"],
        [df["snap_CA"].to_numpy(), df["snap_TX"].to_numpy(), df["snap_WI"].to_numpy()],
        default=0,
    ).astype("int8")
    return df.drop(columns=["snap_CA", "snap_TX", "snap_WI"])


def _price_features(df: pd.DataFrame, prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Weekly price features, computed on the weekly grid then broadcast to days."""
    spec = cfg["features"]
    p = prices.sort_values(["store_id", "item_id", "wm_yr_wk"]).copy()
    grp = p.groupby(["store_id", "item_id"], observed=True, sort=False)["sell_price"]

    p["price_change_pct"] = (grp.pct_change(1) * 100).astype("float32")
    p["price_momentum_4wk"] = (
        grp.pct_change(int(spec["price_momentum_weeks"])) * 100).astype("float32")
    rel_weeks = int(spec["price_rel_weeks"])
    rolling_mean = (
        grp.rolling(rel_weeks, min_periods=1).mean().reset_index(level=[0, 1], drop=True))
    p["price_rel_52wk"] = (p["sell_price"] / rolling_mean).astype("float32")
    p["price_changed_flag"] = (
        p["price_change_pct"].fillna(0).abs() > 1e-6).astype("int8")
    p["price_change_pct"] = p["price_change_pct"].fillna(0).astype("float32")
    p["price_momentum_4wk"] = p["price_momentum_4wk"].fillna(0).astype("float32")

    keep = ["store_id", "item_id", "wm_yr_wk"] + PRICE_FEATURES
    before = len(df)
    df = df.merge(p[keep], on=["store_id", "item_id", "wm_yr_wk"],
                  how="left", validate="m:1")
    assert len(df) == before, "price join changed row count"
    return df


def _external_features(df: pd.DataFrame, fema: pd.DataFrame, fred: pd.DataFrame,
                       cfg: Config) -> pd.DataFrame:
    """Join FEMA/FRED at the forecast origin (target date - demand_shift days)."""
    shift = int(cfg["features"]["demand_shift"])
    df["origin_date"] = df["date"] - pd.Timedelta(days=shift)

    fema_j = fema.rename(columns={"date": "origin_date"}).copy()
    fema_j["state_id"] = fema_j["state_id"].astype(str)
    before = len(df)
    df = df.merge(fema_j[["state_id", "origin_date"] + FEMA_FEATURES],
                  on=["state_id", "origin_date"], how="left", validate="m:1")
    assert len(df) == before, "FEMA join changed row count"

    fred_j = fred.rename(columns={"date": "origin_date"}).copy()
    fred_j["state_id"] = fred_j["state_id"].astype(str)
    df = df.merge(fred_j[["state_id", "origin_date"] + FRED_FEATURES],
                  on=["state_id", "origin_date"], how="left", validate="m:1")
    assert len(df) == before, "FRED join changed row count"
    return df.drop(columns=["origin_date"])


def _extend_forward(df: pd.DataFrame, calendar: pd.DataFrame,
                    meta_cols: pd.DataFrame, last_d: int) -> pd.DataFrame:
    """Add rows for the forward horizon (no actuals) so the deploy model can score.

    Demand features for these rows are fully determined by real observed sales,
    because every demand feature is shifted at least 28 days and the forward
    window is exactly 28 days long.
    """
    max_cal_d = int(calendar["d"].max())
    if last_d >= max_cal_d:
        return df
    future_d = np.arange(last_d + 1, max_cal_d + 1, dtype="int32")
    series = meta_cols[["item_id", "store_id"]].drop_duplicates()
    grid = series.merge(pd.DataFrame({"d": future_d}), how="cross")
    grid["sales"] = np.nan
    out = pd.concat([df, grid], ignore_index=True)
    return out.sort_values(["item_id", "d"], kind="stable").reset_index(drop=True)


def build_store_features(cfg: Config, store_id: str, calendar: pd.DataFrame,
                         series_meta: pd.DataFrame, prices: pd.DataFrame,
                         fema: pd.DataFrame, fred: pd.DataFrame,
                         keep_from_d: int) -> pd.DataFrame:
    """Full feature matrix for one store (memory is bounded per store)."""
    warmup = int(cfg["features"]["warmup_days"])
    meta = series_meta[series_meta["store_id"] == store_id]

    sales = read_sales_long(cfg, stores=[store_id], columns=["item_id", "d", "sales"])
    sales["item_id"] = sales["item_id"].astype(str)
    sales["store_id"] = store_id
    sales["sales"] = sales["sales"].astype("float32")
    sales = sales.sort_values(["item_id", "d"], kind="stable").reset_index(drop=True)

    last_d = int(sales["d"].max())
    sales = _extend_forward(sales, calendar, meta, last_d)

    sales = sales.merge(
        meta[["item_id", "dept_id", "cat_id", "state_id", "release_d"]],
        on="item_id", how="left", validate="m:1")

    sales = _demand_features(sales, cfg)
    sales = _calendar_features(sales, calendar)
    sales = _price_features(sales, prices, cfg)
    sales = _external_features(sales, fema, fred, cfg)

    # Warm-up rows have undefined demand history and are dropped, never imputed.
    min_d = sales["release_d"] + warmup
    sales = sales[(sales["d"] >= min_d) & (sales["d"] >= keep_from_d)]
    sales = sales.drop(columns=["release_d", "wm_yr_wk"]).reset_index(drop=True)
    return sales


def training_start_d(cfg: Config) -> int:
    """Earliest day the feature table must retain, given folds + training window."""
    window = cfg.train_window_days
    if not window:
        return 1
    starts = [spec["train_end_d"] - window + 1 for spec in cfg.folds.values()]
    return max(1, min(starts))


def build_features(cfg: Config, logger=None) -> dict[str, Any]:
    """Build the feature matrix for every selected store and persist it."""
    calendar = read_processed(cfg, "calendar")
    series_meta = read_processed(cfg, "series_meta")
    prices = read_processed(cfg, "prices_long")
    fema = read_processed(cfg, "fema_model_features")
    fred = read_processed(cfg, "fred_state_daily")

    prices["store_id"] = prices["store_id"].astype(str)
    prices["item_id"] = prices["item_id"].astype(str)

    out_root = cfg.processed_dir / "features"
    if out_root.exists():
        for old in out_root.rglob("*.parquet"):
            old.unlink()
    out_root.mkdir(parents=True, exist_ok=True)

    keep_from = training_start_d(cfg)
    stores = sorted(series_meta["store_id"].unique())
    stats = {"stores": len(stores), "rows": 0, "keep_from_d": keep_from,
             "n_features": len(feature_columns("D"))}

    categories: dict[str, list[str]] = {
        "item_id": sorted(series_meta["item_id"].unique().tolist()),
        "dept_id": sorted(series_meta["dept_id"].unique().tolist()),
        "cat_id": sorted(series_meta["cat_id"].unique().tolist()),
        "store_id": sorted(series_meta["store_id"].unique().tolist()),
        "state_id": sorted(series_meta["state_id"].unique().tolist()),
    }
    for col in ("event_name_1", "event_type_1", "event_name_2", "event_type_2"):
        values = calendar[col].fillna(_NO_EVENT).astype(str).unique().tolist()
        categories[col] = sorted(values)

    for store_id in stores:
        frame = build_store_features(cfg, store_id, calendar, series_meta,
                                     prices, fema, fred, keep_from)
        for col, cats in categories.items():
            if col in frame.columns:
                frame[col] = pd.Categorical(frame[col].astype(str), categories=cats)
        part_dir = out_root / f"store_id={store_id}"
        part_dir.mkdir(parents=True, exist_ok=True)
        frame.drop(columns=["store_id"]).to_parquet(
            part_dir / "part-0.parquet", index=False,
            compression="zstd", compression_level=3)
        stats["rows"] += len(frame)
        if logger:
            logger.info("  features %s: %s rows x %s features",
                        store_id, f"{len(frame):,}", stats["n_features"])
        del frame

    (cfg.processed_dir / "feature_categories.json").write_text(
        json.dumps(categories, indent=2), encoding="utf-8")
    return stats


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_features(cfg: Config, columns: list[str] | None = None,
                  d_min: int | None = None, d_max: int | None = None,
                  stores: list[str] | None = None) -> pd.DataFrame:
    """Load the feature matrix with column pruning and day-range filter pushdown."""
    import pyarrow.dataset as ds

    root = cfg.processed_dir / "features"
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found - run: python scripts/prepare_data.py && "
            "python scripts/train.py")
    dataset = ds.dataset(root, format="parquet", partitioning="hive")

    filt = None
    if d_min is not None:
        filt = ds.field("d") >= d_min
    if d_max is not None:
        clause = ds.field("d") <= d_max
        filt = clause if filt is None else (filt & clause)
    if stores:
        clause = ds.field("store_id").isin(stores)
        filt = clause if filt is None else (filt & clause)

    cols = None
    if columns is not None:
        available = set(dataset.schema.names)
        cols = [c for c in dict.fromkeys(columns) if c in available]

    frame = dataset.to_table(columns=cols, filter=filt).to_pandas()

    cat_path = cfg.processed_dir / "feature_categories.json"
    if cat_path.exists():
        categories = json.loads(cat_path.read_text(encoding="utf-8"))
        for col, cats in categories.items():
            if col in frame.columns:
                frame[col] = pd.Categorical(frame[col].astype(str), categories=cats)
    return frame


def feature_family_map() -> dict[str, str]:
    return dict(FEATURE_FAMILY)
