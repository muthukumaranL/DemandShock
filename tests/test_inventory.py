"""Inventory planning arithmetic - every number must trace to a stated formula."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demandshock import inventory as INV


def test_z_matches_the_normal_quantile():
    assert INV.z_for_service_level(0.95) == pytest.approx(1.6448536, abs=1e-6)
    assert INV.z_for_service_level(0.50) == pytest.approx(0.0, abs=1e-12)
    assert INV.z_for_service_level(0.99) == pytest.approx(2.3263479, abs=1e-6)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_invalid_service_level_is_rejected(bad):
    with pytest.raises(ValueError):
        INV.z_for_service_level(bad)


def test_safety_stock_and_reorder_point_are_exact(cfg):
    path = np.full(28, 2.0)
    out = INV.plan(cfg, path, sigma_daily=3.0, lead_time_days=9, service_level=0.95)
    z = 1.6448536269514722
    assert out["lead_time_demand"] == pytest.approx(18.0)      # 9 days x 2 units
    assert out["sigma_lead_time"] == pytest.approx(3.0 * 3.0)  # 3 x sqrt(9)
    assert out["safety_stock"] == pytest.approx(z * 9.0, rel=1e-9)
    assert out["reorder_point"] == pytest.approx(18.0 + z * 9.0, rel=1e-9)


def test_conservative_bound_assumes_perfectly_correlated_errors(cfg):
    out = INV.plan(cfg, np.full(28, 1.0), sigma_daily=2.0, lead_time_days=16,
                   service_level=0.9)
    assert out["sigma_lead_time"] == pytest.approx(2.0 * 4.0)        # sqrt(16)
    assert out["sigma_lead_time_conservative"] == pytest.approx(2.0 * 16.0)
    assert out["sigma_lead_time_conservative"] > out["sigma_lead_time"]


def test_lead_time_is_clamped_to_the_forecast_horizon(cfg):
    out = INV.plan(cfg, np.full(28, 1.0), sigma_daily=1.0, lead_time_days=45,
                   service_level=0.9)
    assert out["lead_time_days_used"] == 28
    assert out["lead_time_clamped"] is True
    assert out["lead_time_demand"] == pytest.approx(28.0)


def test_planning_outputs_are_absent_until_on_hand_is_supplied(cfg):
    out = INV.plan(cfg, np.full(28, 2.0), sigma_daily=1.0, lead_time_days=7,
                   service_level=0.95)
    assert out["days_of_cover"] is None
    assert out["projected_stockout_day"] is None
    assert out["uncovered_units"] is None, (
        "nothing inventory-dependent may be produced without a user-supplied position")


def test_days_of_cover_and_stockout_day(cfg):
    path = np.full(28, 4.0)
    out = INV.plan(cfg, path, sigma_daily=1.0, lead_time_days=7,
                   service_level=0.95, on_hand_units=10.0)
    assert out["days_of_cover"] == pytest.approx(2.5)
    # cumulative demand 4, 8, 12 -> exceeds 10 on day 3
    assert out["projected_stockout_day"] == 3
    assert out["uncovered_units"] == pytest.approx(28 * 4.0 - 10.0)


def test_no_stockout_when_stock_covers_the_whole_horizon(cfg):
    out = INV.plan(cfg, np.full(28, 1.0), sigma_daily=1.0, lead_time_days=7,
                   service_level=0.95, on_hand_units=1000.0)
    assert out["projected_stockout_day"] is None
    assert out["uncovered_units"] == pytest.approx(0.0)
    assert out["position"] == "above_reorder_point"


def test_safety_stock_curve_increases_with_service_level():
    curve = INV.safety_stock_curve(2.0, 9, [0.8, 0.9, 0.95, 0.99])
    assert curve["safety_stock"].is_monotonic_increasing
    assert float(curve.iloc[2]["safety_stock"]) == pytest.approx(1.6448536 * 2.0 * 3.0,
                                                                 rel=1e-6)


def test_exposure_uses_real_prices_and_separates_direction(cfg):
    calendar = pd.DataFrame({"d": [1, 2, 3], "wm_yr_wk": [11101, 11101, 11101]})
    prices = pd.DataFrame({"store_id": ["S"], "item_id": ["A"],
                           "wm_yr_wk": [11101], "sell_price": [2.5]})
    meta = pd.DataFrame({"item_id": ["A"], "store_id": ["S"], "dept_id": ["D"],
                         "cat_id": ["C"], "state_id": ["CA"]})
    forecasts = pd.DataFrame({
        "item_id": ["A"] * 3, "store_id": ["S"] * 3, "d": [1, 2, 3],
        "y_true": [10.0, 2.0, 5.0], "y_pred": [6.0, 5.0, 5.0],
    })
    residuals = forecasts.copy()
    base = INV.build_inventory_base(cfg, forecasts, residuals, prices, calendar, meta)
    row = base.iloc[0]
    # under-forecast on d1 (4 units), over-forecast on d2 (3 units), exact on d3
    assert row["under_exposure_usd"] == pytest.approx(4 * 2.5)
    assert row["over_exposure_usd"] == pytest.approx(3 * 2.5)
    assert row["actual_revenue_usd"] == pytest.approx(17 * 2.5)
    assert row["total_forecast_28"] == pytest.approx(16.0)


def test_exposure_is_never_labelled_as_measured_lost_revenue():
    """M5 records sales, not demand, so unmet demand is unobservable."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    banned = ("lost revenue", "revenue lost", "actual revenue lost")
    offenders = []
    for path in list(root.glob("app/**/*.py")) + list(root.glob("src/**/*.py")) \
            + list(root.glob("api/**/*.py")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            low = line.lower()
            if any(b in low for b in banned) and "not " not in low and "never" not in low:
                offenders.append(f"{path.name}:{i}")
    assert not offenders, f"forecast error must not be presented as lost revenue: {offenders}"
