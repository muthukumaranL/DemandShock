"""Inventory planning and business impact - transparent, and honest about limits.

WHAT M5 DOES NOT CONTAIN
------------------------
There is no inventory-on-hand, no lead time, no service-level target and no cost
data in M5. This module therefore keeps two things strictly apart:

  1. FORECAST-DERIVED figures computed from real data (expected demand, the
     model's own historical error dispersion, real sell prices, and the dollar
     value of forecast error). These always render.

  2. PLANNING figures that require operating assumptions the user must supply
     (lead time, service level, on-hand units, unit cost). These render only
     after the user enters them, are labelled USER INPUT, and are never
     pre-populated with invented values.

"Estimated Revenue Exposure" dollarises forecast error using real M5 prices. It
is NOT measured lost revenue: M5 records sales, not demand, so unmet demand is
unobservable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import Config

SAFETY_STOCK_FORMULA = "safety_stock = z x sigma_daily x sqrt(lead_time_days)"
ROP_FORMULA = "reorder_point = lead_time_demand + safety_stock"

ASSUMPTIONS = [
    "Daily forecast errors are treated as independent across the lead time, so "
    "lead-time error scales with the square root of lead time. A conservative "
    "perfectly-correlated bound (sigma x lead_time) is reported alongside it.",
    "sigma is the standard deviation of this series' own out-of-sample daily "
    "forecast errors over the backtest folds - not an assumed value.",
    "Lead-time demand is the sum of the model's daily forecasts over the lead "
    "time, so lead time cannot exceed the 28-day forecast horizon.",
    "M5 contains no inventory records; on-hand units, lead time, service level "
    "and unit cost are user-supplied operating assumptions.",
]


# ---------------------------------------------------------------------------
# real, forecast-derived base table
# ---------------------------------------------------------------------------
def build_inventory_base(cfg: Config, forecasts: pd.DataFrame,
                         residuals: pd.DataFrame, prices: pd.DataFrame,
                         calendar: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Per series: expected demand, real error dispersion, and dollar exposure.

    `forecasts` is the holdout window of the selected model (it has actuals, so
    exposure is measurable). `residuals` is the pooled cross-validation residual
    set used for sigma.
    """
    if forecasts.empty:
        return pd.DataFrame()

    week_of_day = dict(zip(calendar["d"], calendar["wm_yr_wk"]))
    frame = forecasts.copy()
    frame["item_id"] = frame["item_id"].astype(str)
    frame["store_id"] = frame["store_id"].astype(str)
    frame["wm_yr_wk"] = frame["d"].map(week_of_day)

    price_lookup = prices.copy()
    price_lookup["item_id"] = price_lookup["item_id"].astype(str)
    price_lookup["store_id"] = price_lookup["store_id"].astype(str)
    frame = frame.merge(price_lookup[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
                        on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="m:1")

    error = frame["y_pred"] - frame["y_true"]
    frame["under_units"] = np.clip(-error, 0, None)     # actual exceeded forecast
    frame["over_units"] = np.clip(error, 0, None)       # forecast exceeded actual
    frame["under_exposure_usd"] = frame["under_units"] * frame["sell_price"].fillna(0)
    frame["over_exposure_usd"] = frame["over_units"] * frame["sell_price"].fillna(0)
    frame["actual_revenue_usd"] = frame["y_true"] * frame["sell_price"].fillna(0)

    agg = (frame.groupby(["item_id", "store_id"], observed=True)
           .agg(total_forecast_28=("y_pred", "sum"),
                total_actual_28=("y_true", "sum"),
                mean_daily_forecast=("y_pred", "mean"),
                under_exposure_usd=("under_exposure_usd", "sum"),
                over_exposure_usd=("over_exposure_usd", "sum"),
                actual_revenue_usd=("actual_revenue_usd", "sum"),
                latest_sell_price=("sell_price", "last"),
                priced_days=("sell_price", "count"))
           .reset_index())

    resid = residuals.copy()
    resid["item_id"] = resid["item_id"].astype(str)
    resid["store_id"] = resid["store_id"].astype(str)
    resid["residual"] = resid["y_true"] - resid["y_pred"]
    sigma = (resid.groupby(["item_id", "store_id"], observed=True)["residual"]
             .agg(sigma_daily_resid="std", residual_days="count").reset_index())
    agg = agg.merge(sigma, on=["item_id", "store_id"], how="left")

    min_days = int(cfg["inventory"]["min_priced_days"])
    min_resid = int(cfg["shock"]["eligibility"]["min_residual_days"])
    agg["eligible"] = (
        (agg["priced_days"] >= min_days)
        & (agg["residual_days"].fillna(0) >= min_resid)
        & agg["sigma_daily_resid"].notna())

    agg["exposure_total_usd"] = agg["under_exposure_usd"] + agg["over_exposure_usd"]
    return agg.merge(meta[["item_id", "store_id", "dept_id", "cat_id", "state_id"]],
                     on=["item_id", "store_id"], how="left")


def daily_forecast_path(forecasts: pd.DataFrame, item_id: str,
                        store_id: str) -> pd.DataFrame:
    path = forecasts[(forecasts["item_id"].astype(str) == item_id)
                     & (forecasts["store_id"].astype(str) == store_id)]
    return path.sort_values("d")


# ---------------------------------------------------------------------------
# user-input planning calculations
# ---------------------------------------------------------------------------
def z_for_service_level(service_level: float) -> float:
    if not (0 < service_level < 1):
        raise ValueError("service_level must be strictly between 0 and 1")
    return float(norm.ppf(service_level))


def plan(cfg: Config, daily_forecast: np.ndarray, sigma_daily: float,
         lead_time_days: int, service_level: float,
         on_hand_units: float | None = None,
         unit_cost: float | None = None,
         unit_price: float | None = None) -> dict[str, Any]:
    """Transparent planning arithmetic. Every output traces to a stated formula."""
    max_lead = int(cfg["inventory"]["max_lead_time_days"])
    if lead_time_days < 1:
        raise ValueError("lead_time_days must be at least 1")
    clamped = min(int(lead_time_days), max_lead)

    path = np.asarray(daily_forecast, dtype="float64")
    if path.size == 0:
        raise ValueError("no forecast path available for this series")
    horizon = path[:clamped]

    z = z_for_service_level(service_level)
    lead_time_demand = float(horizon.sum())
    sigma_lt = float(sigma_daily) * np.sqrt(clamped)
    sigma_lt_conservative = float(sigma_daily) * clamped
    safety_stock = z * sigma_lt
    reorder_point = lead_time_demand + safety_stock

    result: dict[str, Any] = {
        "lead_time_days_requested": int(lead_time_days),
        "lead_time_days_used": clamped,
        "lead_time_clamped": clamped != int(lead_time_days),
        "service_level": float(service_level),
        "z": z,
        "sigma_daily": float(sigma_daily),
        "lead_time_demand": lead_time_demand,
        "sigma_lead_time": sigma_lt,
        "sigma_lead_time_conservative": sigma_lt_conservative,
        "safety_stock": safety_stock,
        "safety_stock_conservative": z * sigma_lt_conservative,
        "reorder_point": reorder_point,
        "reorder_point_conservative": lead_time_demand + z * sigma_lt_conservative,
        "formulas": {"safety_stock": SAFETY_STOCK_FORMULA, "reorder_point": ROP_FORMULA},
        "assumptions": list(ASSUMPTIONS),
        "days_of_cover": None,
        "projected_stockout_day": None,
        "uncovered_units": None,
        "uncovered_value_usd": None,
        "excess_units": None,
        "excess_value_usd": None,
        "position": None,
    }

    if on_hand_units is not None:
        on_hand = float(on_hand_units)
        mean_daily = float(path.mean())
        result["days_of_cover"] = (on_hand / mean_daily) if mean_daily > 0 else None

        cumulative = np.cumsum(path)
        breach = np.flatnonzero(cumulative > on_hand)
        result["projected_stockout_day"] = (
            int(breach[0]) + 1 if breach.size else None)

        horizon_demand = float(cumulative[-1])
        result["uncovered_units"] = max(0.0, horizon_demand - on_hand)
        result["excess_units"] = max(0.0, on_hand - reorder_point)
        if unit_price is not None:
            result["uncovered_value_usd"] = result["uncovered_units"] * float(unit_price)
        if unit_cost is not None:
            result["excess_value_usd"] = result["excess_units"] * float(unit_cost)

        if on_hand < reorder_point:
            result["position"] = "below_reorder_point"
        elif result["excess_units"] > 0:
            result["position"] = "above_reorder_point"
        else:
            result["position"] = "at_reorder_point"
    return result


def safety_stock_curve(sigma_daily: float, lead_time_days: int,
                       service_levels: list[float]) -> pd.DataFrame:
    """Safety stock as a function of the chosen service level."""
    rows = []
    for level in service_levels:
        z = z_for_service_level(level)
        rows.append({
            "service_level": level,
            "z": z,
            "safety_stock": z * float(sigma_daily) * np.sqrt(int(lead_time_days)),
        })
    return pd.DataFrame(rows)
