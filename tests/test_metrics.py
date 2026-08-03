"""Metric correctness against values computed by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demandshock import metrics as M

Y = np.array([3.0, 0.0, 2.0, 5.0])
F = np.array([2.0, 1.0, 2.0, 4.0])


def test_pooled_metrics_match_hand_computation():
    got = M.pooled_metrics(Y, F)
    # errors (f - y) = [-1, +1, 0, -1]; |e| = [1, 1, 0, 1]; sum(y) = 10
    assert got["mae"] == pytest.approx(3 / 4, abs=1e-12)
    assert got["rmse"] == pytest.approx(np.sqrt(3 / 4), abs=1e-12)
    assert got["wape"] == pytest.approx(3 / 10, abs=1e-12)
    assert got["bias"] == pytest.approx(-1 / 10, abs=1e-12)
    # sMAPE terms: 200*1/5=40, 200*1/1=200, 0, 200*1/9=22.2222
    assert got["smape"] == pytest.approx((40 + 200 + 0 + 200 / 9) / 4, abs=1e-9)


def test_smape_treats_both_zero_as_perfect():
    terms = M._smape_terms(np.array([0.0, 0.0]), np.array([0.0, 2.0]))
    assert terms[0] == 0.0, "a zero actual with a zero forecast must score 0, not NaN"
    assert terms[1] == pytest.approx(200.0)


def test_wape_and_bias_are_nan_when_nothing_sold():
    got = M.pooled_metrics(np.zeros(3), np.array([1.0, 2.0, 3.0]))
    assert np.isnan(got["wape"]) and np.isnan(got["bias"])
    assert got["mae"] == pytest.approx(2.0)


def test_rmsse_scale_is_the_naive_one_step_error_over_training():
    sales = pd.DataFrame({
        "item_id": ["A"] * 5,
        "store_id": ["S"] * 5,
        "d": [1, 2, 3, 4, 5],
        "sales": [1.0, 3.0, 2.0, 6.0, 100.0],   # d=5 must be excluded by train_end_d
    })
    scales = M.rmsse_scales(sales, train_end_d=4)
    # diffs over d<=4: 2, -1, 4 -> mean of squares = (4 + 1 + 16) / 3
    assert float(scales["scale"].iloc[0]) == pytest.approx(21 / 3, abs=1e-12)


def test_rmsse_is_nan_for_a_flat_series_and_the_exclusion_is_counted():
    sales = pd.DataFrame({
        "item_id": ["FLAT"] * 4 + ["MOVES"] * 4,
        "store_id": ["S"] * 8,
        "d": [1, 2, 3, 4] * 2,
        "sales": [0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 2.0, 6.0],
    })
    scales = M.rmsse_scales(sales, train_end_d=4)
    frame = pd.DataFrame({
        "item_id": ["FLAT", "MOVES"], "store_id": ["S", "S"],
        "d": [5, 5], "y_true": [0.0, 4.0], "y_pred": [1.0, 3.0],
    })
    per = M.series_rmsse(frame, scales)
    assert np.isnan(per.set_index("item_id").loc["FLAT", "rmsse"])
    assert np.isfinite(per.set_index("item_id").loc["MOVES", "rmsse"])

    summary = M.compute_metrics(frame, scales)
    assert int(summary["n_rmsse_excluded"].iloc[0]) == 1, (
        "flat-series exclusions must be counted, not silently dropped")
    assert int(summary["n_series"].iloc[0]) == 2


def test_rmsse_equals_one_when_error_matches_the_naive_benchmark():
    sales = pd.DataFrame({
        "item_id": ["A"] * 4, "store_id": ["S"] * 4,
        "d": [1, 2, 3, 4], "sales": [0.0, 2.0, 0.0, 2.0],
    })
    scales = M.rmsse_scales(sales, train_end_d=4)   # diffs 2,-2,2 -> scale 4
    frame = pd.DataFrame({
        "item_id": ["A", "A"], "store_id": ["S", "S"], "d": [5, 6],
        "y_true": [0.0, 4.0], "y_pred": [2.0, 2.0],   # squared errors 4, 4 -> mse 4
    })
    per = M.series_rmsse(frame, scales)
    assert float(per["rmsse"].iloc[0]) == pytest.approx(1.0)


def test_group_metrics_pool_errors_not_averages_of_ratios():
    """A big series and a tiny one: pooled WAPE must weight by volume."""
    sales = pd.DataFrame({
        "item_id": ["BIG"] * 3 + ["SMALL"] * 3,
        "store_id": ["S"] * 6, "d": [1, 2, 3] * 2,
        "sales": [100.0, 100.0, 100.0, 1.0, 1.0, 1.0],
    })
    scales = M.rmsse_scales(sales, train_end_d=3)
    frame = pd.DataFrame({
        "item_id": ["BIG", "SMALL"], "store_id": ["S", "S"], "d": [4, 4],
        "y_true": [100.0, 1.0], "y_pred": [90.0, 3.0],
    })
    out = M.compute_metrics(frame, scales)
    # pooled: (10 + 2) / (100 + 1)
    assert float(out["wape"].iloc[0]) == pytest.approx(12 / 101)


def test_metrics_by_horizon_uses_only_the_first_h_steps():
    sales = pd.DataFrame({"item_id": ["A"] * 3, "store_id": ["S"] * 3,
                          "d": [1, 2, 3], "sales": [2.0, 4.0, 2.0]})
    scales = M.rmsse_scales(sales, train_end_d=3)
    frame = pd.DataFrame({
        "item_id": ["A"] * 4, "store_id": ["S"] * 4, "d": [4, 5, 6, 7],
        "step": [1, 2, 3, 4],
        "y_true": [2.0, 2.0, 2.0, 100.0], "y_pred": [2.0, 2.0, 2.0, 0.0],
    })
    out = M.metrics_by_horizon(frame, scales, [3, 4])
    by_h = out.set_index("horizon")["mae"]
    assert by_h[3] == pytest.approx(0.0), "step 4 must not leak into horizon 3"
    assert by_h[4] == pytest.approx(25.0)


def test_forecast_accuracy_is_clamped_at_zero():
    assert M.forecast_accuracy(0.2) == pytest.approx(80.0)
    assert M.forecast_accuracy(1.4) == 0.0, (
        "WAPE above 100% must not produce a negative accuracy")
    assert np.isnan(M.forecast_accuracy(float("nan")))


def test_interval_coverage_counts_both_tails():
    frame = pd.DataFrame({
        "y_true": [1.0, 5.0, 10.0, 3.0],
        "p10": [0.0, 0.0, 0.0, 4.0],
        "p90": [2.0, 4.0, 20.0, 9.0],
    })
    got = M.interval_coverage(frame)
    assert got["n"] == 4
    assert got["coverage"] == pytest.approx(0.5)   # rows 0 and 2 inside
    assert got["above"] == pytest.approx(0.25)     # row 1 above p90
    assert got["below"] == pytest.approx(0.25)     # row 3 below p10


def test_wrmsse_is_never_claimed_anywhere_in_the_codebase():
    """The official M5 weighted aggregate is not implemented, so the name is banned."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in list(root.glob("src/**/*.py")) + list(root.glob("app/**/*.py")) \
            + list(root.glob("api/**/*.py")) + list(root.glob("scripts/**/*.py")):
        text = path.read_text(encoding="utf-8").lower()
        for line in text.splitlines():
            if "wrmsse" in line and "never" not in line and "not implement" not in line \
                    and "banned" not in line and "does not" not in line:
                offenders.append(f"{path.name}: {line.strip()[:80]}")
    assert not offenders, f"WRMSSE claimed without implementing it: {offenders}"
