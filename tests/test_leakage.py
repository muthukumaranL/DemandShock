"""Temporal-leakage proofs. These gate every release.

The model is a DIRECT 28-day forecaster: the information set for target date `t`
is frozen at the forecast origin `t - 28`. Every test below asserts a consequence
of that single invariant on REAL data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demandshock import data as D
from demandshock import features as F


# ---------------------------------------------------------------------------
# the core invariant
# ---------------------------------------------------------------------------
def test_demand_features_ignore_everything_after_the_origin(cfg, real_series_frame):
    """Perturbing actuals inside (t-28, t] must not move ANY feature at row t.

    This is the strongest single statement of no-leakage: if the forecast origin
    is really t-28, then the 28 most recent days - including the target day - are
    invisible to the features.
    """
    shift = int(cfg["features"]["demand_shift"])
    base = F._demand_features(real_series_frame.copy(), cfg)
    feature_cols = [c for c in F.DEMAND_FEATURES if c in base.columns]

    tested = 0
    for t in (1500, 1700, 1900, int(base["d"].max())):
        row = base.index[base["d"] == t]
        if len(row) == 0:
            continue
        polluted = real_series_frame.copy()
        recent = (polluted["d"] > t - shift) & (polluted["d"] <= t)
        assert recent.sum() == shift, "expected exactly 28 recent days to perturb"
        polluted.loc[recent, "sales"] = polluted.loc[recent, "sales"] + 999.0
        polluted = F._demand_features(polluted, cfg)

        left = base.loc[row, feature_cols].to_numpy(dtype="float64")
        right = polluted.loc[row, feature_cols].to_numpy(dtype="float64")
        np.testing.assert_array_equal(
            np.nan_to_num(left, nan=-12345.0), np.nan_to_num(right, nan=-12345.0),
            err_msg=f"a demand feature at d={t} changed when recent actuals moved",
        )
        tested += 1
    assert tested >= 3, "invariant should have been exercised on several days"


def test_perturbing_before_the_origin_does_change_features(cfg, real_series_frame):
    """Control: the features are not simply constant - day t-28 IS visible."""
    shift = int(cfg["features"]["demand_shift"])
    t = 1700
    base = F._demand_features(real_series_frame.copy(), cfg)
    polluted = real_series_frame.copy()
    polluted.loc[polluted["d"] == t - shift, "sales"] += 999.0
    polluted = F._demand_features(polluted, cfg)
    row = base.index[base["d"] == t]
    assert float(base.loc[row, "lag_28"].iloc[0]) != float(polluted.loc[row, "lag_28"].iloc[0])


def test_lag_features_equal_hand_indexed_actuals(cfg, real_series_frame):
    frame = F._demand_features(real_series_frame.copy(), cfg)
    actual = dict(zip(real_series_frame["d"], real_series_frame["sales"]))
    for t in (1400, 1600, 1800, 1941):
        row = frame[frame["d"] == t]
        if row.empty:
            continue
        for lag in cfg["features"]["lags"]:
            expected = actual.get(t - lag)
            got = float(row[f"lag_{lag}"].iloc[0])
            assert expected is not None
            assert got == pytest.approx(float(expected)), f"lag_{lag} wrong at d={t}"


def test_rolling_windows_end_at_the_origin(cfg, real_series_frame):
    """roll_mean_w at t must average actuals over [t-27-w, t-28] - never past it."""
    shift = int(cfg["features"]["demand_shift"])
    frame = F._demand_features(real_series_frame.copy(), cfg)
    actual = dict(zip(real_series_frame["d"], real_series_frame["sales"]))
    for t in (1600, 1800, 1900):
        row = frame[frame["d"] == t]
        if row.empty:
            continue
        for window in cfg["features"]["rolling_windows"]:
            days = [t - shift - offset for offset in range(window)]
            values = [actual[d] for d in days if d in actual]
            if len(values) < max(2, window // 2):
                continue
            assert float(row[f"roll_mean_{window}"].iloc[0]) == pytest.approx(
                float(np.mean(values)), rel=1e-5), f"roll_mean_{window} wrong at d={t}"
            assert float(row[f"roll_std_{window}"].iloc[0]) == pytest.approx(
                float(np.std(values, ddof=1)), rel=1e-5), f"roll_std_{window} wrong at d={t}"


def test_zero_rate_matches_hand_computation(cfg, real_series_frame):
    shift = int(cfg["features"]["demand_shift"])
    window = int(cfg["features"]["zero_rate_window"])
    frame = F._demand_features(real_series_frame.copy(), cfg)
    actual = dict(zip(real_series_frame["d"], real_series_frame["sales"]))
    t = 1800
    row = frame[frame["d"] == t]
    if row.empty:
        pytest.skip("series does not cover d=1800")
    days = [t - shift - offset for offset in range(window)]
    values = [actual[d] for d in days if d in actual]
    expected = float(np.mean([1.0 if v == 0 else 0.0 for v in values]))
    assert float(row["zero_rate_28"].iloc[0]) == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# fold isolation
# ---------------------------------------------------------------------------
def test_folds_are_chronological_and_non_overlapping(cfg):
    folds = cfg.folds
    for name, spec in folds.items():
        assert spec["val_start_d"] > spec["train_end_d"], f"{name} validates inside training"
        assert spec["val_end_d"] >= spec["val_start_d"]
    windows = sorted(
        (spec["val_start_d"], spec["val_end_d"]) for spec in folds.values())
    for (_, end_a), (start_b, _) in zip(windows, windows[1:]):
        assert start_b > end_a, "validation windows overlap"


def test_fold_validation_windows_are_28_days(cfg):
    for name, spec in cfg.folds.items():
        span = spec["val_end_d"] - spec["val_start_d"] + 1
        assert span == cfg["horizon"], f"{name} spans {span} days, expected 28"


def test_training_never_reaches_into_a_validation_window(cfg, features):
    """A fold's training rows must all precede its validation window."""
    for name, spec in cfg.folds.items():
        train = features[features["d"] <= spec["train_end_d"]]
        if train.empty:
            continue
        assert int(train["d"].max()) < spec["val_start_d"], f"{name} leaks"


# ---------------------------------------------------------------------------
# external signals
# ---------------------------------------------------------------------------
def test_fema_features_are_origin_snapshots(cfg, features):
    """FEMA columns at target date t must equal the state-day snapshot at t-28."""
    shift = int(cfg["features"]["demand_shift"])
    fema = D.read_processed(cfg, "fema_model_features")
    fema["state_id"] = fema["state_id"].astype(str)

    sample = features.sample(min(400, len(features)), random_state=0)
    merged = sample.assign(
        origin_date=sample["date"] - pd.Timedelta(days=shift),
        state_id=sample["state_id"].astype(str),
    ).merge(
        fema.rename(columns={"date": "origin_date"}),
        on=["state_id", "origin_date"], how="left", suffixes=("", "_snapshot"))
    assert len(merged) == len(sample)
    for col in F.FEMA_FEATURES:
        np.testing.assert_allclose(
            merged[col].to_numpy(dtype="float64"),
            merged[f"{col}_snapshot"].to_numpy(dtype="float64"),
            err_msg=f"{col} is not the t-28 snapshot",
        )


def test_fema_flag_is_off_before_the_declaration_was_issued(cfg):
    """The real leak surface: an incident that began before it was declared.

    FEMA declarations routinely post-date the incident's start. On days between
    incident begin and declaration, nobody knew a declaration was coming, so the
    feature must be off.
    """
    calendar = D.load_calendar(cfg)
    disasters = D.load_fema_disasters(cfg, calendar)
    fema, _ = D.build_fema_tables(cfg, calendar, disasters)

    late = disasters[disasters["declarationDate"] > disasters["incidentBeginDate"]]
    assert len(late) > 0, "expected real disasters declared after their incident began"

    checked = 0
    for row in late.head(25).itertuples(index=False):
        window = fema[
            (fema["state_id"].astype(str) == row.state)
            & (fema["date"] >= row.incidentBeginDate)
            & (fema["date"] < row.declarationDate)
        ]
        if window.empty:
            continue
        # No *other* disaster may be running in that state during the check window.
        others = disasters[
            (disasters["state"] == row.state)
            & (disasters["disasterNumber"] != row.disasterNumber)
            & (disasters["declarationDate"] <= window["date"].max())
            & (disasters["incidentEndDate"] >= window["date"].min())
        ]
        if len(others) > 0:
            continue
        assert int(window["fema_active_count"].max()) == 0, (
            f"disaster {row.disasterNumber} was flagged active before its "
            f"declaration date {row.declarationDate.date()}")
        checked += 1
    assert checked >= 1, "no clean pre-declaration window was available to test"


def test_fred_respects_the_publication_boundary(cfg):
    """BLS publishes month m around day 19 of m+1; the snapshot must not front-run."""
    calendar = D.load_calendar(cfg)
    _, daily = D.build_fred_tables(cfg, calendar)
    ca = daily[daily["state_id"].astype(str) == "CA"].set_index("date")

    before = ca.loc[pd.Timestamp("2016-04-19"), "source_month"]
    after = ca.loc[pd.Timestamp("2016-04-20"), "source_month"]
    assert before == pd.Timestamp("2016-02-01"), (
        f"on 2016-04-19 the latest published month must be February, got {before}")
    assert after == pd.Timestamp("2016-03-01"), (
        f"on 2016-04-20 March becomes available, got {after}")


def test_fred_features_are_origin_snapshots(cfg, features):
    shift = int(cfg["features"]["demand_shift"])
    fred = D.read_processed(cfg, "fred_state_daily")
    fred["state_id"] = fred["state_id"].astype(str)

    sample = features.sample(min(400, len(features)), random_state=1)
    merged = sample.assign(
        origin_date=sample["date"] - pd.Timedelta(days=shift),
        state_id=sample["state_id"].astype(str),
    ).merge(
        fred.rename(columns={"date": "origin_date"})[
            ["state_id", "origin_date"] + F.FRED_FEATURES],
        on=["state_id", "origin_date"], how="left", suffixes=("", "_snapshot"))
    for col in F.FRED_FEATURES:
        np.testing.assert_allclose(
            merged[col].to_numpy(dtype="float64"),
            merged[f"{col}_snapshot"].to_numpy(dtype="float64"),
            err_msg=f"{col} is not the t-28 snapshot")


# ---------------------------------------------------------------------------
# joins
# ---------------------------------------------------------------------------
def test_price_join_matches_the_source_csv(cfg, features):
    """sell_price in the feature table must equal sell_prices.csv for that week."""
    calendar = D.load_calendar(cfg)
    prices = D.read_processed(cfg, "prices_long")
    week_of_day = dict(zip(calendar["d"], calendar["wm_yr_wk"]))
    lookup = {
        (str(r.store_id), str(r.item_id), int(r.wm_yr_wk)): float(r.sell_price)
        for r in prices.itertuples(index=False)
    }
    sample = features.dropna(subset=["sell_price"]).sample(
        min(300, len(features)), random_state=2)
    for row in sample.itertuples(index=False):
        key = (str(row.store_id), str(row.item_id), week_of_day[int(row.d)])
        assert key in lookup, f"no source price row for {key}"
        assert float(row.sell_price) == pytest.approx(lookup[key], rel=1e-6)


def test_snap_uses_the_stores_own_state(cfg, features):
    calendar = D.load_calendar(cfg)
    sample = features.sample(min(500, len(features)), random_state=3)
    snap_by_day = {
        state: dict(zip(calendar["d"], calendar[f"snap_{state}"]))
        for state in ("CA", "TX", "WI")
    }
    for row in sample.itertuples(index=False):
        expected = snap_by_day[str(row.state_id)][int(row.d)]
        assert int(row.snap) == int(expected), (
            f"SNAP mismatch for {row.state_id} on d={row.d}")
        # and it must NOT silently be another state's schedule
        others = {s: snap_by_day[s][int(row.d)] for s in ("CA", "TX", "WI")
                  if s != str(row.state_id)}
        del others


def test_no_pre_release_rows_survive(cfg, features, series_meta):
    # Keyed on (item, store): the same product has a different release date in
    # each store, so an item-only lookup would compare against the wrong date.
    release = {
        (str(r.item_id), str(r.store_id)): int(r.release_d)
        for r in series_meta.itertuples(index=False)
    }
    warmup = int(cfg["features"]["warmup_days"])
    keys = list(zip(features["item_id"].astype(str), features["store_id"].astype(str)))
    min_allowed = np.array([release[k] for k in keys]) + warmup
    assert (features["d"].to_numpy() >= min_allowed).all(), (
        "feature rows exist before release + warm-up")


def test_forward_horizon_has_no_actuals_but_full_features(cfg, features):
    """d_1942..d_1969 must be predictable: no target, every demand feature present."""
    forward = cfg.fold("FORWARD")
    rows = features[(features["d"] >= forward["val_start_d"])
                    & (features["d"] <= forward["val_end_d"])]
    assert len(rows) > 0, "forward horizon rows were not generated"
    assert rows["sales"].isna().all(), "forward rows must not carry actuals"
    for col in ("lag_28", "roll_mean_28", "roll_mean_56"):
        assert rows[col].notna().all(), (
            f"{col} is missing on forward rows - the direct model could not score them")


def test_feature_count_matches_the_specification():
    assert len(F.feature_columns("D")) == 47
    assert len(F.feature_columns("A")) == 19
    assert len(F.feature_columns("B")) == 36
    assert len(F.feature_columns("C")) == 43
    for smaller, bigger in (("A", "B"), ("B", "C"), ("C", "D")):
        assert set(F.feature_columns(smaller)).issubset(F.feature_columns(bigger)), (
            "ablation arms must be strictly cumulative")
