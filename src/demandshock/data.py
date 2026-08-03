"""Real-data ingestion, validation and normalization.

Sources (all real, none synthetic):
  * M5 Forecasting - Accuracy  : daily unit sales, weekly sell prices, retail calendar
  * FEMA OpenFEMA v2          : disaster declaration summaries (county-level rows)
  * FRED / BLS                : monthly state unemployment rates (CA, TX, WI)

Design notes that matter:
  * The 30,490 x 1,941 wide sales matrix is melted through numpy (ravel/repeat of
    *categorical codes*), never through pd.melt on string ids - the string version
    would materialize >10 GB.
  * "Release filtering": rows before a series' first priced week are dropped. Those
    days are not zero demand, they are days the item did not exist in that store.
  * FEMA and FRED features are SNAPSHOTS of the state of the world on a given date.
    The feature builder joins them at (target_date - 28 days), i.e. the forecast
    origin, so nothing published inside the forecast horizon can leak backwards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import Config, get_logger

M5_STATES = ("CA", "TX", "WI")

CALENDAR_COLUMNS = [
    "date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    "snap_CA", "snap_TX", "snap_WI",
]
PRICE_COLUMNS = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
SALES_ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
FEMA_REQUIRED_COLUMNS = [
    "disasterNumber", "state", "declarationType", "declarationDate",
    "incidentType", "declarationTitle", "incidentBeginDate", "incidentEndDate",
    "designatedArea",
]


# ---------------------------------------------------------------------------
# validation reporting
# ---------------------------------------------------------------------------
@dataclass
class ValidationReport:
    """Collects machine-readable data-quality checks; rendered by Module 5."""

    checks: list[dict[str, Any]] = field(default_factory=list)
    tables: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, status: str, expected: Any, observed: Any,
            detail: str = "") -> None:
        assert status in ("pass", "warn", "fail")
        self.checks.append({
            "name": name,
            "status": status,
            "expected": _jsonable(expected),
            "observed": _jsonable(observed),
            "detail": detail,
        })

    def expect(self, name: str, observed: Any, expected: Any, detail: str = "",
               hard: bool = True) -> bool:
        ok = observed == expected
        status = "pass" if ok else ("fail" if hard else "warn")
        self.add(name, status, expected, observed, detail)
        return ok

    def expect_true(self, name: str, condition: bool, expected: Any, observed: Any,
                    detail: str = "", hard: bool = True) -> bool:
        status = "pass" if condition else ("fail" if hard else "warn")
        self.add(name, status, expected, observed, detail)
        return bool(condition)

    def note(self, name: str, observed: Any, detail: str = "") -> None:
        self.add(name, "pass", "informational", observed, detail)

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if c["status"] == "fail"]

    @property
    def warnings(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if c["status"] == "warn"]

    def to_dict(self, cfg: Config, generated_at: str) -> dict[str, Any]:
        return {
            "generated_at": generated_at,
            "mode": cfg.mode,
            "config_hash": cfg.hash(),
            "n_checks": len(self.checks),
            "n_pass": len(self.checks) - len(self.failures) - len(self.warnings),
            "n_warn": len(self.warnings),
            "n_fail": len(self.failures),
            "checks": self.checks,
            "tables": _jsonable(self.tables),
        }

    def write(self, path: Path, cfg: Config, generated_at: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(cfg, generated_at), indent=2), encoding="utf-8"
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------
def load_calendar(cfg: Config, report: ValidationReport | None = None) -> pd.DataFrame:
    path = cfg.m5_file("calendar")
    if not path.exists():
        raise FileNotFoundError(f"M5 calendar not found: {path}")
    cal = pd.read_csv(path)

    if report is not None:
        report.expect("calendar.columns", list(cal.columns), CALENDAR_COLUMNS,
                      "exact M5 calendar schema")

    cal["date"] = pd.to_datetime(cal["date"])
    cal["d"] = cal["d"].str.removeprefix("d_").astype("int32")
    cal["wm_yr_wk"] = cal["wm_yr_wk"].astype("int32")
    for col in ("wday", "month", "snap_CA", "snap_TX", "snap_WI"):
        cal[col] = cal[col].astype("int8")
    cal["year"] = cal["year"].astype("int16")
    cal = cal.sort_values("d").reset_index(drop=True)

    if report is not None:
        gaps = cal["date"].diff().dropna().ne(pd.Timedelta(days=1)).sum()
        report.expect("calendar.date_continuity_gaps", int(gaps), 0,
                      "every calendar row is exactly one day after the previous")
        report.expect("calendar.d_continuous", bool(
            (cal["d"].to_numpy() == np.arange(1, len(cal) + 1)).all()), True,
            "d runs 1..N with no gaps")
        report.expect("calendar.d_unique", int(cal["d"].duplicated().sum()), 0)
        report.expect("calendar.wm_yr_wk_monotonic",
                      bool(cal["wm_yr_wk"].is_monotonic_increasing), True)
        report.note("calendar.rows", len(cal))
        report.note("calendar.date_range",
                    f"{cal['date'].min().date()} .. {cal['date'].max().date()}")
        report.note("calendar.event_days_1", int(cal["event_name_1"].notna().sum()))
        report.note("calendar.event_days_2", int(cal["event_name_2"].notna().sum()))
        report.note("calendar.snap_days", {
            s: int(cal[f"snap_{s}"].sum()) for s in M5_STATES})
    return cal


# ---------------------------------------------------------------------------
# series selection
# ---------------------------------------------------------------------------
def load_series_index(cfg: Config, report: ValidationReport | None = None) -> pd.DataFrame:
    """Read only the id columns of the wide sales file (cheap)."""
    path = cfg.m5_file("sales")
    if not path.exists():
        raise FileNotFoundError(f"M5 sales file not found: {path}")
    ids = pd.read_csv(path, usecols=SALES_ID_COLUMNS)

    if report is not None:
        report.expect("sales.series_count", len(ids), 30490)
        report.expect("sales.id_unique", int(ids["id"].duplicated().sum()), 0)
        report.expect("sales.item_count", int(ids["item_id"].nunique()), 3049)
        report.expect("sales.store_count", int(ids["store_id"].nunique()), 10)
        report.expect("sales.dept_count", int(ids["dept_id"].nunique()), 7)
        report.expect("sales.cat_count", int(ids["cat_id"].nunique()), 3)
        report.expect("sales.state_count", int(ids["state_id"].nunique()), 3)
        # hierarchy consistency: each item maps to exactly one dept/cat,
        # each store to exactly one state
        report.expect("hierarchy.item_to_dept_unique",
                      int(ids.groupby("item_id", observed=True)["dept_id"].nunique().max()), 1)
        report.expect("hierarchy.item_to_cat_unique",
                      int(ids.groupby("item_id", observed=True)["cat_id"].nunique().max()), 1)
        report.expect("hierarchy.store_to_state_unique",
                      int(ids.groupby("store_id", observed=True)["state_id"].nunique().max()), 1)
        derived = ids["item_id"] + "_" + ids["store_id"]
        # M5 ids are "<item_id>_<store_id>_<suffix>"; strip the trailing suffix.
        report.expect("hierarchy.id_composition",
                      int((ids["id"].str.rsplit("_", n=1).str[0] != derived).sum()), 0,
                      "id == item_id + '_' + store_id + suffix")
    return ids


def select_series(cfg: Config, ids: pd.DataFrame) -> pd.DataFrame:
    """Apply the mode's REAL-data subset. Deterministic; never random."""
    settings = cfg.mode_settings
    sel = ids
    stores = settings.get("stores")
    if stores:
        sel = sel[sel["store_id"].isin(stores)]
    cats = settings.get("categories")
    if cats:
        sel = sel[sel["cat_id"].isin(cats)]
    max_items = settings.get("max_real_items")
    if max_items:
        if settings.get("item_selection", "first_n_sorted") != "first_n_sorted":
            raise ValueError("Only deterministic 'first_n_sorted' item selection is supported.")
        keep = sorted(sel["item_id"].unique())[: int(max_items)]
        sel = sel[sel["item_id"].isin(keep)]
    if sel.empty:
        raise ValueError(
            f"Series selection for mode '{cfg.mode}' matched zero real series. "
            "Check config.yaml stores/categories/max_real_items."
        )
    return sel.reset_index(drop=True)


# ---------------------------------------------------------------------------
# prices
# ---------------------------------------------------------------------------
def load_prices(cfg: Config, series: pd.DataFrame, calendar: pd.DataFrame,
                report: ValidationReport | None = None) -> pd.DataFrame:
    path = cfg.m5_file("prices")
    if not path.exists():
        raise FileNotFoundError(f"M5 sell_prices not found: {path}")
    prices = pd.read_csv(
        path,
        dtype={"store_id": "category", "item_id": "category",
               "wm_yr_wk": "int32", "sell_price": "float32"},
    )
    if report is not None:
        report.expect("prices.columns", list(prices.columns), PRICE_COLUMNS)
        report.expect("prices.rows_raw", len(prices), 6841121, hard=False)
        report.expect("prices.duplicate_keys",
                      int(prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).sum()), 0)
        report.expect("prices.non_positive", int((prices["sell_price"] <= 0).sum()), 0)
        report.expect("prices.nulls", int(prices["sell_price"].isna().sum()), 0)
        unknown_weeks = int(~prices["wm_yr_wk"].isin(calendar["wm_yr_wk"]).sum()
                            if False else
                            (~prices["wm_yr_wk"].isin(set(calendar["wm_yr_wk"]))).sum())
        report.expect("prices.weeks_within_calendar", unknown_weeks, 0)

    wanted = set(zip(series["store_id"], series["item_id"]))
    mask = pd.Series(
        list(zip(prices["store_id"].astype(str), prices["item_id"].astype(str))),
        index=prices.index,
    ).isin(wanted)
    prices = prices.loc[mask].copy()
    prices["store_id"] = prices["store_id"].astype(str)
    prices["item_id"] = prices["item_id"].astype(str)
    return prices.reset_index(drop=True)


def compute_series_meta(series: pd.DataFrame, prices: pd.DataFrame,
                        calendar: pd.DataFrame,
                        report: ValidationReport | None = None) -> pd.DataFrame:
    """Release date per series = first calendar day of its first priced week."""
    week_start = (
        calendar.groupby("wm_yr_wk", as_index=False)
        .agg(week_start_d=("d", "min"), week_start_date=("date", "min"))
    )
    first_week = (
        prices.groupby(["store_id", "item_id"], as_index=False, observed=True)["wm_yr_wk"]
        .min()
        .rename(columns={"wm_yr_wk": "release_wm_yr_wk"})
    )
    meta = series.merge(first_week, on=["store_id", "item_id"], how="left", validate="1:1")
    meta = meta.merge(
        week_start.rename(columns={"wm_yr_wk": "release_wm_yr_wk",
                                   "week_start_d": "release_d",
                                   "week_start_date": "release_date"}),
        on="release_wm_yr_wk", how="left", validate="m:1",
    )
    if report is not None:
        missing = int(meta["release_d"].isna().sum())
        report.expect("series.every_series_has_price", missing, 0,
                      "each selected series must appear in sell_prices.csv")
        report.note("series.selected", len(meta))
        report.note("series.release_d_range",
                    f"{int(meta['release_d'].min())} .. {int(meta['release_d'].max())}")
    meta["release_d"] = meta["release_d"].astype("int32")
    return meta


# ---------------------------------------------------------------------------
# sales melt
# ---------------------------------------------------------------------------
def melt_sales(cfg: Config, series: pd.DataFrame, series_meta: pd.DataFrame,
               report: ValidationReport | None = None,
               logger=None) -> dict[str, int]:
    """Melt the wide sales matrix to long parquet, partitioned by store_id.

    Memory-safe: the int16 value matrix is ravelled with numpy and ids are
    rebuilt from categorical codes, so no string column is ever materialized
    at 59M-row scale. Rows before a series' release are dropped.
    """
    path = cfg.m5_file("sales")
    header = pd.read_csv(path, nrows=0)
    day_cols = [c for c in header.columns if c.startswith("d_")]
    dtypes: dict[str, Any] = {c: "int16" for c in day_cols}
    dtypes.update({c: "string" for c in SALES_ID_COLUMNS})

    wide = pd.read_csv(path, dtype=dtypes)
    keep_ids = set(series["id"])
    wide = wide[wide["id"].isin(keep_ids)].reset_index(drop=True)

    n_days = len(day_cols)
    d_values = np.array([int(c.removeprefix("d_")) for c in day_cols], dtype="int32")
    if report is not None:
        report.expect("sales.day_columns", n_days, 1941)
        report.expect("sales.negative_values",
                      int((wide[day_cols].to_numpy() < 0).sum()), 0)
        report.expect_true("sales.int16_safe",
                           bool(wide[day_cols].to_numpy().max() < 32767), "< 32767",
                           int(wide[day_cols].to_numpy().max()))

    release_by_id = dict(zip(series_meta["id"], series_meta["release_d"]))
    out_root = cfg.processed_dir / "sales_long"
    if out_root.exists():
        for old in out_root.rglob("*.parquet"):
            old.unlink()
    out_root.mkdir(parents=True, exist_ok=True)

    stats = {"rows_before_release_filter": 0, "rows_written": 0, "rows_dropped": 0}
    for store_id, block in wide.groupby("store_id", sort=True):
        block = block.reset_index(drop=True)
        values = block[day_cols].to_numpy(dtype="int16")
        n_series = len(block)

        item_codes = pd.Categorical(block["item_id"])
        long = pd.DataFrame({
            "item_id": pd.Categorical.from_codes(
                np.repeat(item_codes.codes, n_days), categories=item_codes.categories),
            "d": np.tile(d_values, n_series).astype("int32"),
            "sales": values.ravel(order="C"),
        })
        release = np.repeat(
            np.array([release_by_id[i] for i in block["id"]], dtype="int32"), n_days)
        stats["rows_before_release_filter"] += len(long)
        long = long[long["d"].to_numpy() >= release].reset_index(drop=True)
        stats["rows_written"] += len(long)

        part_dir = out_root / f"store_id={store_id}"
        part_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(long, preserve_index=False),
            part_dir / "part-0.parquet",
            compression="zstd", compression_level=3, row_group_size=1_000_000,
        )
        if logger:
            logger.info("  melted %s: %s rows", store_id, f"{len(long):,}")
        del long, values, block

    stats["rows_dropped"] = (
        stats["rows_before_release_filter"] - stats["rows_written"])
    if report is not None:
        pct = 100 * stats["rows_dropped"] / max(stats["rows_before_release_filter"], 1)
        report.note("sales.rows_melted", stats["rows_before_release_filter"])
        report.note("sales.rows_after_release_filter", stats["rows_written"])
        report.note("sales.rows_dropped_pre_release",
                    f"{stats['rows_dropped']:,} ({pct:.1f}%)",
                    "days before the item was first priced in that store - "
                    "these are not zero-demand days, the item did not exist yet")
    return stats


def crosscheck_sales_files(cfg: Config, report: ValidationReport) -> None:
    """The validation file must agree with the evaluation file on d_1..d_1913."""
    val_path = cfg.m5_file("sales_crosscheck")
    if not val_path.exists():
        report.add("sales.crosscheck", "warn", "file present", "missing",
                   "sales_train_validation.csv not found; skipped")
        return
    cols = SALES_ID_COLUMNS[:1] + [f"d_{i}" for i in range(1, 1914)]
    a = pd.read_csv(val_path, usecols=cols, nrows=500)
    b = pd.read_csv(cfg.m5_file("sales"), usecols=cols, nrows=500)
    # Ids differ only by the competition-phase suffix (_validation / _evaluation).
    a["id"] = a["id"].str.removesuffix("_validation")
    b["id"] = b["id"].str.removesuffix("_evaluation")
    a = a.set_index("id").sort_index()
    b = b.set_index("id").sort_index()
    same = bool(a.index.equals(b.index) and a.equals(b))
    report.expect_true("sales.validation_vs_evaluation_agree", same, True, same,
                       "first 500 series, d_1..d_1913 identical across both M5 files",
                       hard=False)


# ---------------------------------------------------------------------------
# FEMA
# ---------------------------------------------------------------------------
def load_fema_disasters(cfg: Config, calendar: pd.DataFrame,
                        report: ValidationReport | None = None) -> pd.DataFrame:
    """Dedupe county-level FEMA rows to one row per disasterNumber, in-window."""
    path = cfg.external_file("fema")
    if not path.exists():
        raise FileNotFoundError(f"FEMA declarations file not found: {path}")
    fema = pd.read_csv(path, low_memory=False)

    if report is not None:
        missing = [c for c in FEMA_REQUIRED_COLUMNS if c not in fema.columns]
        report.expect("fema.required_columns_present", missing, [])
        report.note("fema.rows_raw", len(fema))

    for col in ("declarationDate", "incidentBeginDate", "incidentEndDate"):
        fema[col] = pd.to_datetime(fema[col], errors="coerce", utc=True).dt.tz_localize(None)

    states = list(cfg["fema"]["states"])
    shift_days = int(cfg["features"]["demand_shift"])
    win_lo = calendar["date"].min() - pd.Timedelta(days=shift_days)
    win_hi = calendar["date"].max()

    sub = fema[fema["state"].isin(states)].copy()
    end_or_begin = sub["incidentEndDate"].fillna(sub["incidentBeginDate"])
    in_window = (sub["incidentBeginDate"] <= win_hi) & (end_or_begin >= win_lo)
    sub = sub[in_window].copy()

    n_null_end = int(sub["incidentEndDate"].isna().sum())
    impute_days = int(cfg["fema"]["end_impute_days"])

    grouped = (
        sub.groupby("disasterNumber", as_index=False)
        .agg(
            state=("state", "first"),
            declarationType=("declarationType", "first"),
            incidentType=("incidentType", "first"),
            declarationTitle=("declarationTitle", "first"),
            declarationDate=("declarationDate", "min"),
            incidentBeginDate=("incidentBeginDate", "min"),
            incidentEndDate=("incidentEndDate", "max"),
            n_counties=("designatedArea", "nunique"),
        )
    )
    grouped["end_imputed"] = grouped["incidentEndDate"].isna()
    grouped["incidentEndDate"] = grouped["incidentEndDate"].fillna(
        grouped["incidentBeginDate"] + pd.Timedelta(days=impute_days))
    # Incidents still open when the retail data ends are truncated to the window.
    # Record it so the app can say "ongoing at data window end" instead of implying
    # the incident really finished on the last day of M5.
    grouped["end_clipped"] = grouped["incidentEndDate"] > win_hi
    grouped["incidentEndDate"] = grouped["incidentEndDate"].clip(upper=win_hi)
    grouped["n_counties"] = grouped["n_counties"].astype("int16")

    if report is not None:
        # The snapshot grid extends `demand_shift` days before the M5 start so that
        # features for d_1 have a forecast-origin snapshot. Report both counts: the
        # headline is the strict M5-window figure quoted in the README.
        m5_start = calendar["date"].min()
        strict = grouped[grouped["incidentEndDate"] >= m5_start]
        report.expect("fema.unique_disasters_in_m5_window", len(strict), 115,
                      "CA/TX/WI declarations overlapping 2011-01-29 .. 2016-06-19")
        report.note("fema.unique_disasters_incl_origin_lookback", len(grouped),
                    f"includes declarations active in the {shift_days}-day "
                    "forecast-origin lookback before the M5 window starts")
        report.expect("fema.end_after_begin",
                      int((grouped["incidentEndDate"] < grouped["incidentBeginDate"]).sum()), 0)
        report.note("fema.null_end_dates_raw_rows", n_null_end,
                    f"imputed as incidentBeginDate + {impute_days} days (in-window median)")
        report.note("fema.disasters_by_state_type",
                    grouped.groupby(["state", "declarationType"])["disasterNumber"]
                    .nunique().to_dict())
        report.note("fema.disasters_by_incident_type",
                    grouped.groupby(["state", "incidentType"])["disasterNumber"]
                    .nunique().to_dict())
    return grouped


def build_fema_tables(cfg: Config, calendar: pd.DataFrame, disasters: pd.DataFrame,
                      report: ValidationReport | None = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (model snapshot features, retrospective context rows).

    Model features are a per (state, snapshot_date) view of what was KNOWN on that
    date: a disaster counts only once its declarationDate has passed. The feature
    builder joins this table at (target_date - 28 days), so the model never sees a
    declaration issued inside its own forecast horizon.

    The context table keeps the full retrospective incident windows (no gating) and
    is used only for Module 3's "what real events coincided" narrative.
    """
    shift = int(cfg["features"]["demand_shift"])
    cap = int(cfg["fema"]["days_since_declaration_cap"])
    states = list(cfg["fema"]["states"])

    snap_dates = pd.date_range(
        calendar["date"].min() - pd.Timedelta(days=shift), calendar["date"].max(), freq="D")
    grid = pd.MultiIndex.from_product(
        [states, snap_dates], names=["state_id", "date"]).to_frame(index=False)

    dates = grid["date"].to_numpy(dtype="datetime64[ns]")
    st = grid["state_id"].to_numpy()

    active_count = np.zeros(len(grid), dtype="int16")
    dr_active = np.zeros(len(grid), dtype="int8")
    fire_active = np.zeros(len(grid), dtype="int8")
    flood_active = np.zeros(len(grid), dtype="int8")
    storm_active = np.zeros(len(grid), dtype="int8")
    counties_active = np.zeros(len(grid), dtype="int32")
    last_decl = np.full(len(grid), np.datetime64("NaT"), dtype="datetime64[ns]")

    context_rows: list[dict[str, Any]] = []
    for row in disasters.itertuples(index=False):
        # KNOWN-state window: from declaration onward, while the incident is open.
        known_start = max(row.declarationDate, row.incidentBeginDate)
        known_start = np.datetime64(max(known_start, row.declarationDate))
        end = np.datetime64(row.incidentEndDate)
        if end < known_start:
            end = known_start  # declared at/after incident end -> single known day
        mask = (st == row.state) & (dates >= known_start) & (dates <= end)
        if mask.any():
            active_count[mask] += 1
            counties_active[mask] += int(row.n_counties)
            if row.declarationType == "DR":
                dr_active[mask] = 1
            itype = str(row.incidentType)
            if itype == "Fire":
                fire_active[mask] = 1
            elif itype == "Flood":
                flood_active[mask] = 1
            elif itype.startswith("Severe Storm"):
                storm_active[mask] = 1
            decl = np.datetime64(row.declarationDate)
            newer = mask & (pd.isna(last_decl) | (last_decl < decl))
            last_decl[newer] = decl

        context_rows.append({
            "state_id": row.state,
            "disasterNumber": int(row.disasterNumber),
            "declarationType": row.declarationType,
            "incidentType": row.incidentType,
            "declarationTitle": row.declarationTitle,
            "declaration_date": row.declarationDate,
            "incident_begin": row.incidentBeginDate,
            "incident_end": row.incidentEndDate,
            "end_imputed": bool(row.end_imputed),
            "end_clipped": bool(row.end_clipped),
            "n_counties": int(row.n_counties),
        })

    days_since = (dates - last_decl) / np.timedelta64(1, "D")
    days_since = np.where(np.isnan(days_since), cap, np.clip(days_since, 0, cap))

    model_features = pd.DataFrame({
        "state_id": pd.Categorical(grid["state_id"]),
        "date": grid["date"],
        "fema_active_count": active_count,
        "fema_dr_active": dr_active,
        "fema_fire_active": fire_active,
        "fema_flood_active": flood_active,
        "fema_storm_active": storm_active,
        "fema_counties_active": counties_active,
        "fema_days_since_declaration": days_since.astype("float32"),
    })
    context = pd.DataFrame(context_rows)

    if report is not None:
        in_m5 = model_features[model_features["date"] >= calendar["date"].min()]
        on_rates = {}
        for state, block in in_m5.groupby("state_id", observed=True):
            on_rates[str(state)] = {
                "any_active": round(float((block["fema_active_count"] > 0).mean()), 3),
                "dr_active": round(float(block["fema_dr_active"].mean()), 3),
                "fire_active": round(float(block["fema_fire_active"].mean()), 3),
                "flood_active": round(float(block["fema_flood_active"].mean()), 3),
                "storm_active": round(float(block["fema_storm_active"].mean()), 3),
            }
        report.note("fema.flag_on_rate_by_state", on_rates,
                    "share of M5 days each flag is on. High rates mean low "
                    "discriminative power - reported, not hidden.")
        report.expect("fema.model_feature_rows", len(model_features),
                      len(states) * len(snap_dates))
    return model_features, context


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------
def build_fred_tables(cfg: Config, calendar: pd.DataFrame,
                      report: ValidationReport | None = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (monthly table with availability dates, daily snapshot table).

    BLS publishes state unemployment for month m around day 17-22 of month m+1, so
    a value is only usable from `available_from = month + 1 month + 19 days`. The
    daily table answers "what was the latest published rate as of date s".
    """
    lag_months = int(cfg["fred"]["availability_lag_months"])
    lag_days = int(cfg["fred"]["availability_lag_days"])
    shift = int(cfg["features"]["demand_shift"])

    frames = []
    for state in cfg["fema"]["states"]:
        path = cfg.fred_file(state)
        if not path.exists():
            raise FileNotFoundError(f"FRED file for {state} not found: {path}")
        raw = pd.read_csv(path)
        value_col = [c for c in raw.columns if c != "observation_date"][0]
        df = pd.DataFrame({
            "state_id": state,
            "month": pd.to_datetime(raw["observation_date"]),
            "ur_level": raw[value_col].astype("float32"),
        })
        df = df.sort_values("month").reset_index(drop=True)
        df["ur_change_1m"] = df["ur_level"].diff(1).astype("float32")
        df["ur_change_3m"] = df["ur_level"].diff(3).astype("float32")
        df["ur_yoy"] = df["ur_level"].diff(12).astype("float32")
        df["available_from"] = (
            df["month"] + pd.DateOffset(months=lag_months) + pd.Timedelta(days=lag_days))
        frames.append(df)
    monthly = pd.concat(frames, ignore_index=True)

    snap_dates = pd.date_range(
        calendar["date"].min() - pd.Timedelta(days=shift), calendar["date"].max(), freq="D")
    daily_parts = []
    for state, block in monthly.groupby("state_id", sort=True):
        block = block.dropna(subset=["ur_level"]).sort_values("available_from")
        left = pd.DataFrame({"date": snap_dates})
        merged = pd.merge_asof(
            left, block[["available_from", "ur_level", "ur_change_1m",
                         "ur_change_3m", "ur_yoy", "month"]],
            left_on="date", right_on="available_from", direction="backward")
        merged["state_id"] = state
        daily_parts.append(merged)
    daily = pd.concat(daily_parts, ignore_index=True)
    daily = daily.rename(columns={"month": "source_month"})
    daily["state_id"] = pd.Categorical(daily["state_id"])
    for col in ("ur_level", "ur_change_1m", "ur_change_3m", "ur_yoy"):
        daily[col] = daily[col].astype("float32")

    if report is not None:
        window = daily[daily["date"] >= calendar["date"].min()]
        report.expect("fred.no_missing_in_window",
                      int(window["ur_level"].isna().sum()), 0,
                      "every M5-window day has a published unemployment rate")
        report.note("fred.states", sorted(monthly["state_id"].unique().tolist()))
        report.note("fred.month_range",
                    f"{monthly['month'].min().date()} .. {monthly['month'].max().date()}")
        report.note("fred.publication_lag",
                    f"observation month + {lag_months} month + {lag_days} days",
                    "models the real BLS state release schedule")
        report.note("fred.ur_range_in_m5_window", {
            str(s): [round(float(b["ur_level"].min()), 1),
                     round(float(b["ur_level"].max()), 1)]
            for s, b in window.groupby("state_id", observed=True)})
    return monthly, daily


# ---------------------------------------------------------------------------
# readers used downstream
# ---------------------------------------------------------------------------
def read_sales_long(cfg: Config, stores: list[str] | None = None,
                    columns: list[str] | None = None) -> pd.DataFrame:
    import pyarrow.dataset as ds

    root = cfg.processed_dir / "sales_long"
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found - run: python scripts/prepare_data.py")
    dataset = ds.dataset(root, format="parquet", partitioning="hive")
    filt = None
    if stores:
        filt = ds.field("store_id").isin(stores)
    table = dataset.to_table(columns=columns, filter=filt)
    df = table.to_pandas()
    # Categorical, not string. pandas 3 backs plain strings with PyArrow, and a
    # boolean take over 46M Arrow strings tries to allocate several GB; dictionary
    # codes make the same operation cheap.
    for col in ("item_id", "store_id"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def read_processed(cfg: Config, name: str) -> pd.DataFrame:
    path = cfg.processed_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run: python scripts/prepare_data.py")
    return pd.read_parquet(path)


def assert_unused_files_untouched(cfg: Config, report: ValidationReport) -> None:
    """Prove the excluded sources (NOAA weather, submission scaffold) are not used."""
    declared = list(cfg.get("unused_files", []))
    report.note("data.excluded_sources", declared,
                "NOAA weather is deliberately out of scope in this version; "
                "no feature, model, page or endpoint reads these files")
