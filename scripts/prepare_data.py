"""Stage 1 - ingest, validate and normalize the real source data.

Reads   : M5 (calendar, sales, sell_prices), FEMA declarations, FRED unemployment
Writes  : data/processed/*.parquet  +  artifacts/validation_report.json

Run:  python scripts/prepare_data.py [--mode development|full]
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
import pandas as pd

from demandshock import data as D
from demandshock import features as F
from demandshock.config import load_config, get_logger


def main() -> int:
    parser = argparse.ArgumentParser(description="DemandShock data preparation")
    parser.add_argument("--mode", choices=["development", "full"], default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, mode=args.mode)
    cfg.ensure_dirs()
    log = get_logger("prepare_data", cfg, "prepare_data.log")
    started = time.time()
    log.info("=" * 68)
    log.info("DemandShock | data preparation | mode=%s | config=%s",
             cfg.mode, cfg.hash())
    log.info("=" * 68)

    report = D.ValidationReport()
    D.assert_unused_files_untouched(cfg, report)

    # --- calendar --------------------------------------------------------
    log.info("[1/7] calendar")
    calendar = D.load_calendar(cfg, report)
    calendar.to_parquet(cfg.processed_dir / "calendar.parquet", index=False)
    log.info("      %s rows, %s .. %s", f"{len(calendar):,}",
             calendar["date"].min().date(), calendar["date"].max().date())

    # --- series selection -------------------------------------------------
    log.info("[2/7] series index + selection")
    ids = D.load_series_index(cfg, report)
    series = D.select_series(cfg, ids)
    log.info("      selected %s of %s real series (%s stores, %s items)",
             f"{len(series):,}", f"{len(ids):,}",
             series["store_id"].nunique(), series["item_id"].nunique())
    D.crosscheck_sales_files(cfg, report)

    # --- prices + release dates ------------------------------------------
    log.info("[3/7] sell prices + release dates")
    prices = D.load_prices(cfg, series, calendar, report)
    series_meta = D.compute_series_meta(series, prices, calendar, report)
    prices.to_parquet(cfg.processed_dir / "prices_long.parquet", index=False)
    series_meta.to_parquet(cfg.processed_dir / "series_meta.parquet", index=False)
    log.info("      %s price rows for the selected series", f"{len(prices):,}")

    # --- sales melt -------------------------------------------------------
    log.info("[4/7] melting wide sales -> long parquet (release-filtered)")
    stats = D.melt_sales(cfg, series, series_meta, report, logger=log)
    log.info("      %s rows written (%s dropped as pre-release)",
             f"{stats['rows_written']:,}", f"{stats['rows_dropped']:,}")

    # --- measured series descriptors --------------------------------------
    # M5 ships no product names, so each series is described by attributes read
    # from its own history rather than by an invented label.
    sales_for_labels = D.read_sales_long(cfg, columns=["item_id", "store_id", "sales"])
    series_meta = D.build_series_labels(series_meta, sales_for_labels, prices, report)
    series_meta.to_parquet(cfg.processed_dir / "series_meta.parquet", index=False)
    del sales_for_labels
    log.info("      described %s series by department, price band and sales velocity",
             f"{len(series_meta):,}")

    # --- FEMA -------------------------------------------------------------
    log.info("[5/7] FEMA declarations -> state-day tables")
    disasters = D.load_fema_disasters(cfg, calendar, report)
    fema_features, fema_context = D.build_fema_tables(cfg, calendar, disasters, report)
    fema_features.to_parquet(cfg.processed_dir / "fema_model_features.parquet", index=False)
    fema_context.to_parquet(cfg.processed_dir / "fema_context.parquet", index=False)
    log.info("      %s unique disasters -> %s state-day snapshot rows",
             len(disasters), f"{len(fema_features):,}")

    # --- FRED -------------------------------------------------------------
    log.info("[6/7] FRED unemployment -> availability-dated tables")
    fred_monthly, fred_daily = D.build_fred_tables(cfg, calendar, report)
    fred_monthly.to_parquet(cfg.processed_dir / "fred_state_monthly.parquet", index=False)
    fred_daily.to_parquet(cfg.processed_dir / "fred_state_daily.parquet", index=False)
    log.info("      %s monthly observations -> %s daily snapshot rows",
             f"{len(fred_monthly):,}", f"{len(fred_daily):,}")

    # --- feature matrix ---------------------------------------------------
    log.info("[7/8] feature engineering (origin-frozen, 28-day information set)")
    feat_stats = F.build_features(cfg, logger=log)
    log.info("      %s feature rows from d>=%s across %s stores",
             f"{feat_stats['rows']:,}", feat_stats["keep_from_d"], feat_stats["stores"])
    report.note("features.rows", feat_stats["rows"])
    report.note("features.count", feat_stats["n_features"],
                "47 canonical features; ablation arms A/B/C/D are cumulative subsets")
    report.note("features.first_day_retained", feat_stats["keep_from_d"],
                "earliest day needed by the configured folds and training window")
    report.note("features.warmup_days_dropped", cfg["features"]["warmup_days"],
                "28-day shift + 56-day longest rolling window; never imputed")

    # --- summary tables for the Data Quality tab --------------------------
    log.info("[8/8] validation report")
    sales_long = D.read_sales_long(cfg, columns=["item_id", "d", "sales"])
    zero_ratio = float((sales_long["sales"] == 0).mean())
    report.note("sales.zero_demand_ratio", round(zero_ratio, 4),
                "share of post-release days with zero units sold (intermittent demand)")
    report.note("sales.mean_units_per_day", round(float(sales_long["sales"].mean()), 3))
    report.tables = {
        "mode": cfg.mode,
        "series": int(len(series_meta)),
        "items": int(series_meta["item_id"].nunique()),
        "stores": int(series_meta["store_id"].nunique()),
        "departments": int(series_meta["dept_id"].nunique()),
        "categories": int(series_meta["cat_id"].nunique()),
        "states": int(series_meta["state_id"].nunique()),
        "calendar_days": int(len(calendar)),
        "date_start": calendar["date"].min(),
        "date_end": calendar["date"].max(),
        "sales_rows": int(len(sales_long)),
        "price_rows": int(len(prices)),
        "fema_disasters": int(len(disasters)),
        "fred_months": int(len(fred_monthly)),
        "zero_demand_ratio": round(zero_ratio, 4),
    }

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report.write(cfg.artifacts_dir / "validation_report.json", cfg, generated_at)
    report.write(cfg.artifacts_dir / "data_quality.json", cfg, generated_at)

    elapsed = time.time() - started
    log.info("-" * 68)
    for check in report.failures:
        log.error("FAIL  %s: expected %s, observed %s",
                  check["name"], check["expected"], check["observed"])
    for check in report.warnings:
        log.warning("WARN  %s: expected %s, observed %s",
                    check["name"], check["expected"], check["observed"])
    log.info("%s checks: %s pass, %s warn, %s fail | %.1fs",
             len(report.checks),
             len(report.checks) - len(report.failures) - len(report.warnings),
             len(report.warnings), len(report.failures), elapsed)

    if report.failures:
        log.error("Data preparation FAILED. Fix the source data or config and re-run.")
        return 1
    log.info("Data preparation complete -> %s", cfg.processed_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
