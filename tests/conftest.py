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
def features(cfg, processed_available):
    """The real development-mode feature matrix."""
    if not processed_available:
        pytest.skip("processed features not built; run scripts/prepare_data.py")
    return F.load_features(cfg)


@pytest.fixture(scope="session")
def sales_long(cfg, processed_available):
    if not processed_available:
        pytest.skip("processed data not built; run scripts/prepare_data.py")
    return D.read_sales_long(cfg)


@pytest.fixture(scope="session")
def series_meta(cfg, processed_available):
    if not processed_available:
        pytest.skip("processed data not built; run scripts/prepare_data.py")
    return D.read_processed(cfg, "series_meta")


@pytest.fixture(scope="session")
def busy_series(sales_long) -> str:
    """A real item with plenty of non-zero demand - best signal for feature tests."""
    totals = sales_long.groupby("item_id", observed=True)["sales"].sum()
    return str(totals.idxmax())


@pytest.fixture(scope="session")
def real_series_frame(cfg, sales_long, series_meta, busy_series) -> pd.DataFrame:
    """One real series, contiguous daily, ready for feature-function unit tests."""
    frame = sales_long[sales_long["item_id"] == busy_series].copy()
    frame["item_id"] = frame["item_id"].astype(str)
    frame["sales"] = frame["sales"].astype("float32")
    meta = series_meta[series_meta["item_id"] == busy_series].iloc[0]
    frame["release_d"] = int(meta["release_d"])
    return frame.sort_values("d").reset_index(drop=True)
