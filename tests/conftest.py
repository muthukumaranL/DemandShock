"""Shared pytest fixtures.

Fixtures are slices of the REAL local datasets, extracted at session start. Nothing
is committed to the repository (M5 competition terms restrict redistribution) and
no fixture data ever reaches a model, a dashboard or a reported result - these are
software unit-test inputs only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from demandshock.config import load_config  # noqa: E402
from demandshock import data as D  # noqa: E402
from demandshock import features as F  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(REPO_ROOT / "config.yaml", mode="development")


@pytest.fixture(scope="session")
def raw_available(cfg) -> bool:
    return cfg.m5_file("calendar").exists() and cfg.m5_file("sales").exists()


@pytest.fixture(scope="session")
def processed_available(cfg) -> bool:
    return (cfg.processed_dir / "features").exists()


@pytest.fixture(scope="session")
def calendar(cfg, raw_available):
    if not raw_available:
        pytest.skip("raw M5 data not present")
    return D.load_calendar(cfg)


@pytest.fixture(scope="session")
def test_store(cfg, processed_available) -> str:
    """One real store. Tests are per-series, so a single partition is enough and
    keeps them fast whether the pipeline last ran in development or full mode."""
    if not processed_available:
        pytest.skip("processed data not built; run scripts/prepare_data.py")
    partitions = sorted((cfg.processed_dir / "features").glob("store_id=*"))
    if not partitions:
        pytest.skip("no feature partitions found")
    return partitions[0].name.split("=", 1)[1]


@pytest.fixture(scope="session")
def features(cfg, processed_available, test_store):
    """Real feature rows for one store (full mode holds 25M+ rows in total)."""
    if not processed_available:
        pytest.skip("processed features not built; run scripts/prepare_data.py")
    frame = F.load_features(cfg, stores=[test_store])
    if "store_id" not in frame.columns:
        frame = frame.assign(store_id=test_store)
    return frame


@pytest.fixture(scope="session")
def sales_long(cfg, processed_available, test_store):
    if not processed_available:
        pytest.skip("processed data not built; run scripts/prepare_data.py")
    return D.read_sales_long(cfg, stores=[test_store])


@pytest.fixture(scope="session")
def series_meta(cfg, processed_available):
    if not processed_available:
        pytest.skip("processed data not built; run scripts/prepare_data.py")
    return D.read_processed(cfg, "series_meta")


@pytest.fixture(scope="session")
def busy_series(sales_long, test_store) -> tuple[str, str]:
    """A real item-store series with plenty of non-zero demand.

    Keyed by BOTH ids: an item_id alone is not a series - the same product exists
    in up to ten stores with different release dates and demand histories.
    """
    totals = sales_long.groupby("item_id", observed=True)["sales"].sum()
    return str(totals.idxmax()), test_store


@pytest.fixture(scope="session")
def real_series_frame(cfg, sales_long, series_meta, busy_series) -> pd.DataFrame:
    """One real series, contiguous daily, ready for feature-function unit tests."""
    item_id, store_id = busy_series
    frame = sales_long[sales_long["item_id"].astype(str) == item_id].copy()
    frame["item_id"] = frame["item_id"].astype(str)
    frame["store_id"] = store_id
    frame["sales"] = frame["sales"].astype("float32")
    meta = series_meta[(series_meta["item_id"].astype(str) == item_id)
                       & (series_meta["store_id"].astype(str) == store_id)].iloc[0]
    frame["release_d"] = int(meta["release_d"])
    assert not frame["d"].duplicated().any(), "fixture must be a single series"
    return frame.sort_values("d").reset_index(drop=True)
