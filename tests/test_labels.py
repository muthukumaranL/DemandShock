"""Series descriptors must be derived from real data, never invented.

M5 anonymises product identities: the source contains no product names. Instead of
inventing them, every series carries a description measured from its own history.
These tests pin that the descriptors are genuinely derived, are stable, and never
leak into the model as features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demandshock import data as D
from demandshock import features as F

LABEL_COLUMNS = ["total_units", "mean_daily", "zero_share", "median_price",
                 "velocity_rank", "price_rank", "velocity", "price_band", "dept_label"]


@pytest.fixture(scope="module")
def labelled(cfg, processed_available):
    if not processed_available:
        pytest.skip("processed data not built; run scripts/prepare_data.py")
    meta = D.read_processed(cfg, "series_meta")
    if "dept_label" not in meta.columns:
        pytest.skip("artifacts predate series descriptors")
    return meta


def test_m5_really_contains_no_product_names(cfg, raw_available):
    """The premise: if M5 ever shipped names, deriving descriptors would be wrong."""
    if not raw_available:
        pytest.skip("raw M5 data not present")
    sales_cols = pd.read_csv(cfg.m5_file("sales"), nrows=0).columns
    id_cols = [c for c in sales_cols if not c.startswith("d_")]
    assert id_cols == ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    price_cols = list(pd.read_csv(cfg.m5_file("prices"), nrows=0).columns)
    assert price_cols == ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    for column in id_cols + price_cols:
        assert "name" not in column.lower(), (
            f"{column} looks like a product name; descriptors would be unnecessary")


def test_every_series_is_described(labelled):
    for column in LABEL_COLUMNS:
        assert column in labelled.columns, f"missing descriptor column {column}"
    assert labelled["dept_label"].notna().all()
    assert labelled["velocity"].notna().all()
    assert labelled["price_band"].notna().all()


def test_descriptor_vocabularies_are_the_configured_ones(labelled):
    assert set(labelled["velocity"]).issubset(
        {name for _, name in D.VELOCITY_BANDS} | {"Unranked"})
    assert set(labelled["price_band"]).issubset(
        {name for _, name in D.PRICE_BANDS} | {"Unpriced"})


def test_department_label_is_derived_from_the_real_hierarchy(labelled):
    assert D._pretty_dept("FOODS_3") == "Foods - Dept 3"
    assert D._pretty_dept("HOUSEHOLD_1") == "Household - Dept 1"
    for row in labelled.sample(min(200, len(labelled)), random_state=0).itertuples():
        assert row.dept_label == D._pretty_dept(row.dept_id)


def test_velocity_matches_measured_sales_volume(cfg, labelled, test_store):
    """A 'Top seller' must really be in the top decile of its category by units."""
    sales = D.read_sales_long(cfg, stores=[test_store], columns=["item_id", "sales"])
    totals = sales.groupby("item_id", observed=True)["sales"].sum()
    block = labelled[labelled["store_id"].astype(str) == test_store]
    for row in block.sample(min(150, len(block)), random_state=1).itertuples():
        measured = float(totals.get(str(row.item_id), np.nan))
        assert float(row.total_units) == pytest.approx(measured), (
            "total_units must equal the series' real summed sales")

    top = block[block["velocity"] == "Top seller"]
    slow = block[block["velocity"] == "Very slow"]
    if len(top) and len(slow):
        assert top["total_units"].min() > slow["total_units"].max(), (
            "velocity bands must be monotone in measured volume")


def test_price_band_matches_measured_median_price(cfg, labelled, test_store):
    prices = D.read_processed(cfg, "prices_long")
    prices = prices[prices["store_id"].astype(str) == test_store]
    medians = prices.groupby("item_id", observed=True)["sell_price"].median()
    block = labelled[labelled["store_id"].astype(str) == test_store]
    for row in block.sample(min(150, len(block)), random_state=2).itertuples():
        expected = float(medians.get(str(row.item_id), np.nan))
        assert float(row.median_price) == pytest.approx(expected, rel=1e-5)

    premium = block[block["price_band"] == "Premium"]
    value = block[block["price_band"] == "Value"]
    if len(premium) and len(value):
        assert premium["median_price"].min() > value["median_price"].max(), (
            "price bands must be monotone in measured price")


def test_ranks_are_computed_within_category(labelled):
    """A slow hobby item must not be judged against a grocery staple."""
    for category, block in labelled.groupby("cat_id", observed=True):
        if len(block) < 20:
            continue
        assert block["velocity_rank"].max() == pytest.approx(1.0, abs=1e-6), (
            f"category {category} should contain its own top-ranked series")


def test_measured_inputs_are_reproducible(cfg, labelled, test_store):
    """The measured quantities behind the descriptors must rebuild exactly.

    Only the measured inputs are checked here, not the bands. Ranks are taken
    across every store in a category (see build_series_labels), so recomputing
    them from a single store's rows would legitimately place series differently -
    that is the design, not drift.
    """
    sales = D.read_sales_long(cfg, stores=[test_store],
                              columns=["item_id", "store_id", "sales"])
    prices = D.read_processed(cfg, "prices_long")
    base = labelled.drop(columns=LABEL_COLUMNS)
    base = base[base["store_id"].astype(str) == test_store]
    rebuilt = D.build_series_labels(base, sales, prices)

    original = labelled[labelled["store_id"].astype(str) == test_store]
    merged = original.merge(rebuilt, on=["item_id", "store_id"], suffixes=("", "_new"))
    assert len(merged) == len(original)
    for column in ("total_units", "median_price", "dept_label"):
        if column == "dept_label":
            assert (merged[column] == merged[f"{column}_new"]).all()
        else:
            np.testing.assert_allclose(
                merged[column].to_numpy(dtype="float64"),
                merged[f"{column}_new"].to_numpy(dtype="float64"), rtol=1e-5,
                err_msg=f"{column} did not reproduce from the same rows")


def test_banding_is_a_pure_deterministic_function():
    """Given a rank, the band must be fixed - no randomness, no drift."""
    for rank in (0.0, 0.14, 0.15, 0.39, 0.40, 0.69, 0.70, 0.89, 0.90, 1.0):
        assert D._band(rank, D.VELOCITY_BANDS) == D._band(rank, D.VELOCITY_BANDS)
    assert D._band(1.00, D.VELOCITY_BANDS) == "Top seller"
    assert D._band(0.90, D.VELOCITY_BANDS) == "Top seller"
    assert D._band(0.89, D.VELOCITY_BANDS) == "Fast mover"
    assert D._band(0.00, D.VELOCITY_BANDS) == "Very slow"
    assert D._band(0.80, D.PRICE_BANDS) == "Premium"
    assert D._band(0.79, D.PRICE_BANDS) == "Mid-price"
    assert D._band(0.00, D.PRICE_BANDS) == "Value"


def test_descriptors_are_not_model_features():
    """Display labels must never influence a forecast."""
    features = set(F.feature_columns("D"))
    for column in LABEL_COLUMNS:
        assert column not in features, (
            f"{column} is a display descriptor and must not be a model feature")


def test_descriptors_never_invent_a_product_name(labelled):
    """Guard against a future edit slipping fabricated names into the pipeline."""
    invented = ("milk", "bread", "shampoo", "cereal", "soda", "widget", "brand",
                "deluxe", "premium organic", "product ")
    for column in ("dept_label", "velocity", "price_band"):
        joined = " ".join(labelled[column].astype(str).unique()).lower()
        for word in invented:
            assert word not in joined, (
                f"{column} contains {word!r}, which is not derivable from M5")
