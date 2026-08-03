"""Provenance: processed tables must match the RAW source CSVs.

Every other test in this suite reads `data/processed/`. That leaves one gap: if the
wide-to-long melt or the price ingest were systematically wrong, the error would
propagate identically into features, artifacts and tests, and everything would still
pass. These tests close the loop by going back to the original Kaggle CSVs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demandshock import data as D

# Day columns sampled across the whole history: early, middle, late, and the final day.
SAMPLE_DAYS = (150, 800, 1500, 1941)


@pytest.fixture(scope="module")
def raw_sales_slice(cfg, raw_available, test_store, series_meta):
    """Raw wide rows for the selected series of one real store."""
    if not raw_available:
        pytest.skip("raw M5 data not present")
    columns = ["id", "item_id", "store_id"] + [f"d_{d}" for d in SAMPLE_DAYS]
    raw = pd.read_csv(cfg.m5_file("sales"), usecols=columns)
    wanted = set(series_meta[series_meta["store_id"] == test_store]["item_id"].astype(str))
    raw = raw[(raw["store_id"] == test_store) & (raw["item_id"].isin(wanted))]
    assert not raw.empty, "no raw rows matched the processed selection"
    return raw


def test_melted_sales_match_the_raw_csv(raw_sales_slice, sales_long, series_meta,
                                        test_store):
    """Post-release values must equal the raw file exactly, cell for cell."""
    melted = {
        (str(r.item_id), int(r.d)): float(r.sales)
        for r in sales_long.itertuples(index=False)
    }
    release = {
        str(r.item_id): int(r.release_d)
        for r in series_meta[series_meta["store_id"] == test_store].itertuples(index=False)
    }

    checked = 0
    for row in raw_sales_slice.itertuples(index=False):
        item = str(row.item_id)
        for day in SAMPLE_DAYS:
            if day < release[item]:
                continue
            raw_value = float(getattr(row, f"d_{day}"))
            assert (item, day) in melted, (
                f"{item} d_{day} is post-release but missing from sales_long")
            assert melted[(item, day)] == raw_value, (
                f"{item} d_{day}: melted {melted[(item, day)]} != raw {raw_value}")
            checked += 1
    assert checked >= 100, f"only {checked} post-release cells were comparable"


def test_pre_release_rows_are_dropped_and_were_zero_in_the_raw_file(
        raw_sales_slice, sales_long, series_meta, test_store):
    """Release filtering must remove only days the item genuinely did not sell on.

    If a dropped day carried non-zero units in the raw file, the filter would be
    discarding real demand rather than pre-assortment padding.
    """
    melted = {(str(r.item_id), int(r.d)) for r in sales_long.itertuples(index=False)}
    release = {
        str(r.item_id): int(r.release_d)
        for r in series_meta[series_meta["store_id"] == test_store].itertuples(index=False)
    }

    checked = 0
    for row in raw_sales_slice.itertuples(index=False):
        item = str(row.item_id)
        for day in SAMPLE_DAYS:
            if day >= release[item]:
                continue
            assert (item, day) not in melted, (
                f"{item} d_{day} precedes release but survived into sales_long")
            assert float(getattr(row, f"d_{day}")) == 0.0, (
                f"{item} d_{day} was dropped as pre-release but sold "
                f"{getattr(row, f'd_{day}')} units in the raw file")
            checked += 1
    if checked == 0:
        pytest.skip("no sampled day fell before a release date for this store")


def test_processed_prices_match_the_raw_csv(cfg, raw_available, test_store):
    """sell_price in prices_long must equal sell_prices.csv for the same key."""
    if not raw_available:
        pytest.skip("raw M5 data not present")
    prices = D.read_processed(cfg, "prices_long")
    prices = prices[prices["store_id"].astype(str) == test_store]
    if prices.empty:
        pytest.skip("no processed prices for the test store")

    raw = pd.read_csv(
        cfg.m5_file("prices"),
        dtype={"store_id": "string", "item_id": "string",
               "wm_yr_wk": "int32", "sell_price": "float64"})
    raw = raw[raw["store_id"] == test_store]
    lookup = {
        (str(r.item_id), int(r.wm_yr_wk)): float(r.sell_price)
        for r in raw.itertuples(index=False)
    }

    sample = prices.sample(min(200, len(prices)), random_state=0)
    for row in sample.itertuples(index=False):
        key = (str(row.item_id), int(row.wm_yr_wk))
        assert key in lookup, f"no raw price row for {test_store} {key}"
        # Prices are stored as float32 to keep the 6.8M-row table small, so compare
        # at that precision rather than demanding float64 equality.
        assert np.float32(row.sell_price) == np.float32(lookup[key]), (
            f"{key}: processed {row.sell_price} != raw {lookup[key]}")


def test_fema_context_matches_the_raw_declaration_file(cfg, raw_available):
    """Dedupe must preserve the real incident span and county count per disaster."""
    if not raw_available:
        pytest.skip("raw data not present")
    context_path = cfg.processed_dir / "fema_context.parquet"
    if not context_path.exists():
        pytest.skip("fema_context not built")
    context = pd.read_parquet(context_path)

    raw = pd.read_csv(cfg.external_file("fema"), low_memory=False)
    for column in ("incidentBeginDate", "incidentEndDate"):
        raw[column] = pd.to_datetime(
            raw[column], errors="coerce", utc=True).dt.tz_localize(None)

    calendar = D.read_processed(cfg, "calendar")
    window_end = calendar["date"].max()

    checked = 0
    for row in context.sample(min(20, len(context)), random_state=1).itertuples(index=False):
        source = raw[raw["disasterNumber"] == row.disasterNumber]
        assert not source.empty, f"disaster {row.disasterNumber} absent from raw file"
        assert row.incident_begin == source["incidentBeginDate"].min()
        assert int(row.n_counties) == int(source["designatedArea"].nunique()), (
            "county count must be the distinct designated areas, not the row count")

        raw_end = source["incidentEndDate"].max()
        if row.end_clipped:
            assert row.incident_end == window_end, (
                "a clipped end must sit exactly on the data window end")
            assert pd.isna(raw_end) or raw_end > window_end
        elif not row.end_imputed:
            assert row.incident_end == raw_end
        checked += 1
    assert checked > 0


def test_clipped_incident_ends_are_flagged_not_silently_truncated(cfg):
    """DR-4272 (TX, ends 2016-06-24) must be marked as ongoing, not simply shortened."""
    context_path = cfg.processed_dir / "fema_context.parquet"
    if not context_path.exists():
        pytest.skip("fema_context not built")
    context = pd.read_parquet(context_path)
    assert "end_clipped" in context.columns, (
        "the context table must record whether an incident end was truncated")

    calendar = D.read_processed(cfg, "calendar")
    window_end = calendar["date"].max()
    clipped = context[context["end_clipped"]]
    for row in clipped.itertuples(index=False):
        assert row.incident_end == window_end
