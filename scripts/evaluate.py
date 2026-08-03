"""Stage 3 - demand-shock analysis and inventory/business-impact artifacts.

Consumes the backtest residuals produced by scripts/train.py and writes:
  artifacts/shocks_daily.parquet, shock_episodes.parquet, inventory_base.parquet

Run:  python scripts/evaluate.py [--mode development|full]
"""

from __future__ import annotations

import argparse
import json
import time

import _bootstrap  # noqa: F401
import pandas as pd

from demandshock import data as D
from demandshock import inventory as INV
from demandshock import shock as S
from demandshock.config import load_config, get_logger


def main() -> int:
    parser = argparse.ArgumentParser(description="DemandShock evaluation & shock analysis")
    parser.add_argument("--mode", choices=["development", "full"], default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, mode=args.mode)
    cfg.ensure_dirs()
    log = get_logger("evaluate", cfg, "evaluate.log")
    started = time.time()

    residual_path = cfg.artifacts_dir / "shock_residuals.parquet"
    if not residual_path.exists():
        log.error("%s not found - run scripts/train.py first", residual_path)
        return 1

    metadata_path = cfg.artifacts_dir / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = metadata["selected_config"]
    shock_config = metadata["shock_config"]

    log.info("=" * 68)
    log.info("DemandShock | shock + inventory analysis | mode=%s", cfg.mode)
    log.info("shock residual source: arm %s (no FEMA/FRED features by design)",
             shock_config)
    log.info("=" * 68)

    calendar = D.read_processed(cfg, "calendar")
    meta = D.read_processed(cfg, "series_meta")
    meta["item_id"] = meta["item_id"].astype(str)
    meta["store_id"] = meta["store_id"].astype(str)
    prices = D.read_processed(cfg, "prices_long")
    fema_context = D.read_processed(cfg, "fema_context")
    fred_daily = D.read_processed(cfg, "fred_state_daily")

    residuals = pd.read_parquet(residual_path)

    # Shock analysis needs recent actuals only: the longest lookback is the 56-day
    # eligibility gate before the first scored day, plus rolling baselines. Loading
    # the full history would cost 46M rows in full mode for no benefit.
    history_buffer = 90
    sales_from = int(residuals["d"].min()) - history_buffer
    sales = D.read_sales_long(cfg, columns=["item_id", "d", "sales"])
    sales = sales[sales["d"] >= sales_from]
    sales["item_id"] = sales["item_id"].astype(str)
    sales = sales.merge(meta[["item_id", "store_id"]], on="item_id", how="left")
    sales["sales"] = sales["sales"].astype("float64")
    log.info("loaded %s actual item-days from d_%s for shock baselines",
             f"{len(sales):,}", sales_from)
    day_to_date = dict(zip(calendar["d"], calendar["date"]))
    log.info("[1/3] scoring daily shock signals over d_%s..d_%s (%s .. %s)",
             int(residuals["d"].min()), int(residuals["d"].max()),
             day_to_date[int(residuals["d"].min())].date(),
             day_to_date[int(residuals["d"].max())].date())

    daily = S.score_daily(cfg, residuals, sales, logger=log)
    if daily.empty:
        log.error("No series were eligible for shock scoring.")
        return 1
    daily["date"] = daily["d"].map(day_to_date)
    daily = daily.merge(meta[["item_id", "store_id", "dept_id", "cat_id", "state_id"]],
                        on=["item_id", "store_id"], how="left")
    daily.to_parquet(cfg.artifacts_dir / "shocks_daily.parquet", index=False)

    band_counts = daily["band"].value_counts().to_dict()
    log.info("    %s scored series-days | band mix: %s",
             f"{len(daily):,}", json.dumps({k: int(v) for k, v in band_counts.items()}))

    log.info("[2/3] grouping episodes and classifying")
    episodes = S.build_episodes(cfg, daily, calendar, meta, sales)
    episodes = S.attach_context(cfg, episodes, fema_context, fred_daily)
    episodes.to_parquet(cfg.artifacts_dir / "shock_episodes.parquet", index=False)
    if episodes.empty:
        log.warning("    no episodes met the filing threshold")
    else:
        log.info("    %s episodes | %s", len(episodes),
                 json.dumps(episodes["classification"].value_counts().to_dict()))
        log.info("    severity: %s",
                 json.dumps(episodes["band"].value_counts().to_dict()))
        overlap = float(episodes["fema_overlap"].mean())
        log.info("    %.1f%% of episodes coincide with an active FEMA declaration "
                 "in the same state (coincidence, not causation)", 100 * overlap)

    log.info("[3/3] inventory base table")
    holdout_dir = (cfg.artifacts_dir / "forecasts.parquet"
                   / f"model=lgbm/config={selected}/fold_id=HOLDOUT")
    if not holdout_dir.exists():
        log.error("holdout forecasts missing at %s", holdout_dir)
        return 1
    holdout = pd.read_parquet(holdout_dir)

    cv_parts = []
    for fold_name in cfg.cv_folds:
        part = (cfg.artifacts_dir / "forecasts.parquet"
                / f"model=lgbm/config={selected}/fold_id={fold_name}")
        if part.exists():
            cv_parts.append(pd.read_parquet(part))
    cv_residuals = pd.concat(cv_parts, ignore_index=True) if cv_parts else holdout

    base = INV.build_inventory_base(cfg, holdout, cv_residuals, prices, calendar, meta)
    base.to_parquet(cfg.artifacts_dir / "inventory_base.parquet", index=False)
    log.info("    %s series | %s eligible for planning | "
             "under-exposure $%s, over-exposure $%s over the 28-day holdout",
             f"{len(base):,}", f"{int(base['eligible'].sum()):,}",
             f"{base['under_exposure_usd'].sum():,.0f}",
             f"{base['over_exposure_usd'].sum():,.0f}")

    log.info("-" * 68)
    log.info("Evaluation complete in %.1fs -> %s", time.time() - started, cfg.artifacts_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
