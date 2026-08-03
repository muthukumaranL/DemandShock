"""Stage 2 - train, backtest, ablate, and produce every forecasting artifact.

Runs, in order:
  1. objective spot-check (development mode, fold 1, arm B) - then frozen
  2. ablation A/B/C/D over the rolling-origin folds F1..F3
  3. naive / seasonal-naive benchmarks over the same folds
  4. mechanical feature-set selection, then a single holdout fit (opened once)
  5. an extra arm-B holdout fit whose residuals feed shock detection
  6. the deploy fit through d_1941 and its forward forecast
  7. empirical residual quantiles -> P10/P90

Writes: artifacts/forecasts.parquet (partitioned), metrics.parquet, ablation.*,
objective_comparison.csv, residual_quantiles.parquet, feature_importance.parquet,
model_metadata.json, fold_definitions.json, models/*.txt

Run:  python scripts/train.py [--mode development|full]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
import lightgbm as lgb
import numpy as np
import pandas as pd

from demandshock import data as D
from demandshock import features as F
from demandshock import forecasting as FC
from demandshock import metrics as M
from demandshock.config import load_config, get_logger

HIERARCHY_LEVELS = {
    "overall": [],
    "state": ["state_id"],
    "store": ["store_id"],
    "category": ["cat_id"],
    "department": ["dept_id"],
}


def _forecast_dir(cfg):
    return cfg.artifacts_dir / "forecasts.parquet"


def _write_forecasts(cfg, frame: pd.DataFrame, model: str, config_name: str,
                     fold_id: str) -> None:
    """One hive partition per (model, config, fold) so pages load only their slice."""
    if frame.empty:
        return
    part = (_forecast_dir(cfg) / f"model={model}" / f"config={config_name}"
            / f"fold_id={fold_id}")
    part.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    for col in ("item_id", "store_id"):
        out[col] = out[col].astype(str)
    out.to_parquet(part / "part-0.parquet", index=False,
                   compression="zstd", compression_level=3)


def _attach_hierarchy(frame: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    cols = ["item_id", "store_id", "dept_id", "cat_id", "state_id"]
    return frame.merge(meta[cols], on=["item_id", "store_id"], how="left")


def _metric_rows(frame: pd.DataFrame, scales: pd.DataFrame, meta: pd.DataFrame,
                 cfg, model: str, config_name: str, fold_id: str) -> pd.DataFrame:
    """Metrics at every hierarchy level and horizon, plus a daily series."""
    if frame.empty or frame["y_true"].notna().sum() == 0:
        return pd.DataFrame()
    enriched = _attach_hierarchy(frame, meta)
    rows = []
    for level, group_cols in HIERARCHY_LEVELS.items():
        block = M.metrics_by_horizon(enriched, scales, cfg["horizons"], group_cols or None)
        if block.empty:
            continue
        block["level"] = level
        block["group_key"] = (
            "ALL" if not group_cols
            else block[group_cols[0]].astype(str))
        rows.append(block.drop(columns=group_cols))

    daily = M.compute_metrics(enriched, scales, ["d"])
    if not daily.empty:
        daily["level"] = "day"
        daily["group_key"] = daily["d"].astype(int).astype(str)
        daily["horizon"] = pd.NA
        rows.append(daily.drop(columns=["d"]))

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "model", model)
    out.insert(1, "config", config_name)
    out.insert(2, "fold_id", fold_id)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="DemandShock training & backtesting")
    parser.add_argument("--mode", choices=["development", "full"], default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-objective-check", action="store_true")
    parser.add_argument("--ablation-configs", default=None,
                        help="comma-separated subset, e.g. B for a smoke run "
                             "(reduced runs must not be published as ablation results)")
    args = parser.parse_args()

    cfg = load_config(args.config, mode=args.mode)
    cfg.ensure_dirs()
    log = get_logger("train", cfg, "train.log")
    started = time.time()

    ablation_configs = (
        [c.strip() for c in args.ablation_configs.split(",")]
        if args.ablation_configs else list(cfg["ablation"]["configs"]))
    reduced_run = ablation_configs != list(cfg["ablation"]["configs"])

    log.info("=" * 68)
    log.info("DemandShock | training | mode=%s | configs=%s | config_hash=%s",
             cfg.mode, ",".join(ablation_configs), cfg.hash())
    log.info("=" * 68)

    meta = D.read_processed(cfg, "series_meta")
    meta["item_id"] = meta["item_id"].astype(str)
    meta["store_id"] = meta["store_id"].astype(str)
    sales = D.read_sales_long(cfg, columns=["item_id", "d", "sales"])
    sales["item_id"] = sales["item_id"].astype(str)
    sales = sales.merge(meta[["item_id", "store_id"]], on="item_id", how="left")
    sales["sales"] = sales["sales"].astype("float64")

    if _forecast_dir(cfg).exists():
        shutil.rmtree(_forecast_dir(cfg))

    all_metrics: list[pd.DataFrame] = []
    ablation_rows: list[dict] = []
    scales_cache: dict[str, pd.DataFrame] = {}
    residual_pool: list[pd.DataFrame] = []
    shock_residuals: list[pd.DataFrame] = []
    boosters: dict[tuple[str, str], FC.TrainedModel] = {}

    def scales_for(fold: FC.FoldSpec) -> pd.DataFrame:
        if fold.name not in scales_cache:
            scales_cache[fold.name] = M.rmsse_scales(sales, fold.train_end_d)
        return scales_cache[fold.name]

    # ---------------------------------------------------------------- baselines
    log.info("[1/6] naive benchmarks")
    for fold in FC.fold_specs(cfg, cfg.eval_folds):
        actuals = sales[(sales["d"] >= fold.val_start_d) & (sales["d"] <= fold.val_end_d)]
        for model in FC.BASELINE_MODELS:
            preds = FC.baseline_forecast(sales, fold, model)
            frame = preds.merge(
                actuals.rename(columns={"sales": "y_true"})[
                    ["item_id", "store_id", "d", "y_true"]],
                on=["item_id", "store_id", "d"], how="inner")
            _write_forecasts(cfg, frame, model, "na", fold.name)
            block = _metric_rows(frame, scales_for(fold), meta, cfg,
                                 model, "na", fold.name)
            if not block.empty:
                all_metrics.append(block)
        log.info("    %s: benchmarks scored on %s series",
                 fold.name, f"{actuals['item_id'].nunique():,}")

    # ------------------------------------------------- objective spot-check
    objective_path = cfg.artifacts_dir / "objective_comparison.csv"
    if cfg.is_dev and not args.skip_objective_check:
        log.info("[2/6] objective spot-check (dev, fold F1, arm B) - then frozen")
        fold = FC.fold_specs(cfg, ["F1"])[0]
        frame = F.load_features(cfg, d_max=fold.val_end_d)
        rows = []
        for objective in ("tweedie", "poisson", "regression"):
            override = {"objective": objective}
            if objective != "tweedie":
                override["tweedie_variance_power"] = None
                override.pop("tweedie_variance_power")
                override = {"objective": objective}
            params = cfg.lgbm_params()
            params.update(override)
            if objective != "tweedie":
                params.pop("tweedie_variance_power", None)
            model = FC.train_lgbm(cfg, frame, fold, "B", params_override=params,
                                  logger=log)
            preds = FC.predict_window(model, frame, fold)
            stats = M.pooled_metrics(preds["y_true"].to_numpy(), preds["y_pred"].to_numpy())
            per = M.series_rmsse(preds, scales_for(fold))
            rows.append({
                "objective": objective, "fold": fold.name, "horizon": 28,
                "wape": stats["wape"], "rmsse": float(per["rmsse"].mean(skipna=True)),
                "bias": stats["bias"], "best_iteration": model.best_iteration,
                "train_seconds": round(model.train_seconds, 1),
            })
            log.info("    %-12s WAPE=%.4f  RMSSE=%.4f  bias=%+.4f",
                     objective, stats["wape"], rows[-1]["rmsse"], stats["bias"])
        pd.DataFrame(rows).to_csv(objective_path, index=False)
        del frame
    else:
        log.info("[2/6] objective spot-check skipped (frozen: tweedie, p=1.1)")

    # ------------------------------------------------------------- ablation
    log.info("[3/6] ablation %s over folds %s",
             "/".join(ablation_configs), ",".join(cfg.cv_folds))
    for fold in FC.fold_specs(cfg, cfg.cv_folds):
        window = cfg.train_window_days
        d_min = (fold.train_end_d - window + 1) if window else None
        frame = F.load_features(cfg, d_min=d_min, d_max=fold.val_end_d)
        for config_name in ablation_configs:
            model = FC.train_lgbm(cfg, frame, fold, config_name, logger=log)
            preds = FC.predict_window(model, frame, fold)
            _write_forecasts(cfg, preds, "lgbm", config_name, fold.name)

            block = _metric_rows(preds, scales_for(fold), meta, cfg,
                                 "lgbm", config_name, fold.name)
            if not block.empty:
                all_metrics.append(block)
                for horizon in cfg["horizons"]:
                    row = block[(block["level"] == "overall")
                                & (block["horizon"] == horizon)]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    ablation_rows.append({
                        "config": config_name, "fold_id": fold.name, "horizon": horizon,
                        **{m: float(r[m]) for m in M.METRIC_NAMES},
                        "n_series": int(r["n_series"]),
                        "n_rmsse_excluded": int(r["n_rmsse_excluded"]),
                        "n_features": len(F.feature_columns(config_name)),
                        "train_rows": model.train_rows,
                        "best_iteration": model.best_iteration,
                        "train_seconds": round(model.train_seconds, 1),
                    })
            if config_name == cfg["ablation"]["shock_config"]:
                shock_residuals.append(preds.assign(fold_id=fold.name))
            boosters[(config_name, fold.name)] = model
        del frame

    ablation = pd.DataFrame(ablation_rows)
    if not ablation.empty:
        ablation["config_hash"] = cfg.hash()
        ablation["mode"] = cfg.mode
        ablation["reduced_run"] = reduced_run
        ablation.to_parquet(cfg.artifacts_dir / "ablation.parquet", index=False)
        ablation.to_csv(cfg.artifacts_dir / "ablation.csv", index=False)

    # ------------------------------------------------------------- selection
    selection_metric = cfg["ablation"]["selection_metric"]
    selection_h = int(cfg["ablation"]["selection_horizon"])
    tolerance = float(cfg["ablation"]["selection_tolerance_rel"])
    if ablation.empty:
        selected = ablation_configs[-1]
        selection_detail = {"rule": "no ablation rows; fell back to last config"}
    else:
        means = (ablation[ablation["horizon"] == selection_h]
                 .groupby("config")[selection_metric].mean().sort_index())
        best_value = float(means.min())
        threshold = best_value * (1 + tolerance)
        eligible = [c for c in ["A", "B", "C", "D"]
                    if c in means.index and float(means[c]) <= threshold]
        selected = eligible[0] if eligible else str(means.idxmin())
        selection_detail = {
            "rule": f"smallest feature set within {tolerance:.1%} relative of the "
                    f"best mean {selection_metric.upper()} at h={selection_h}",
            "mean_by_config": {c: round(float(v), 6) for c, v in means.items()},
            "best_value": round(best_value, 6),
            "threshold": round(threshold, 6),
            "selected": selected,
        }
        log.info("    selection: %s", json.dumps(selection_detail["mean_by_config"]))
    log.info("    selected feature set: %s (%s)", selected,
             F.ABLATION_LABELS.get(selected, ""))

    # ------------------------------------------------------- holdout + shock
    log.info("[4/6] holdout fits (opened once)")
    holdout = FC.fold_specs(cfg, ["HOLDOUT"])[0]
    window = cfg.train_window_days
    d_min = (holdout.train_end_d - window + 1) if window else None
    frame = F.load_features(cfg, d_min=d_min, d_max=holdout.val_end_d)

    holdout_model = FC.train_lgbm(cfg, frame, holdout, selected, logger=log)
    holdout_preds = FC.predict_window(holdout_model, frame, holdout)
    block = _metric_rows(holdout_preds, scales_for(holdout), meta, cfg,
                         "lgbm", selected, holdout.name)
    if not block.empty:
        all_metrics.append(block)

    shock_config = cfg["ablation"]["shock_config"]
    if shock_config == selected:
        shock_holdout_preds = holdout_preds.copy()
        shock_holdout_model = holdout_model
    else:
        log.info("    extra %s fit for shock residuals (not used for selection)",
                 shock_config)
        shock_holdout_model = FC.train_lgbm(cfg, frame, holdout, shock_config, logger=log)
        shock_holdout_preds = FC.predict_window(shock_holdout_model, frame, holdout)
        blk = _metric_rows(shock_holdout_preds, scales_for(holdout), meta, cfg,
                           "lgbm", shock_config, holdout.name)
        if not blk.empty:
            all_metrics.append(blk)
    shock_residuals.append(shock_holdout_preds.assign(fold_id=holdout.name))

    # residual pool for uncertainty: selected config, cross-validation folds only
    for fold_name in cfg.cv_folds:
        key = (selected, fold_name)
        if key in boosters:
            fold = FC.fold_specs(cfg, [fold_name])[0]
            part = pd.read_parquet(
                _forecast_dir(cfg) / f"model=lgbm/config={selected}/fold_id={fold_name}")
            residual_pool.append(part)

    quantiles = pd.DataFrame()
    if residual_pool:
        pooled = pd.concat(residual_pool, ignore_index=True)
        quantiles = FC.fit_residual_quantiles(cfg, pooled, meta)
        quantiles.to_parquet(cfg.artifacts_dir / "residual_quantiles.parquet", index=False)
        log.info("    residual quantiles from %s out-of-sample residuals (%s groups)",
                 f"{len(pooled):,}", len(quantiles))

    if not quantiles.empty:
        holdout_preds = FC.apply_residual_quantiles(cfg, holdout_preds, quantiles, meta)
    _write_forecasts(cfg, holdout_preds, "lgbm", selected, holdout.name)
    if shock_config != selected:
        _write_forecasts(cfg, shock_holdout_preds, "lgbm", shock_config, holdout.name)

    coverage = M.interval_coverage(holdout_preds) if "p10" in holdout_preds else {}
    if coverage:
        log.info("    holdout P10-P90 coverage: %.1f%% (target ~80%%) on %s points",
                 100 * coverage["coverage"], f"{coverage['n']:,}")

    # ----------------------------------------------------- deploy + forward
    log.info("[5/6] deploy fit through d_%s + forward forecast", cfg.fold("FORWARD")["train_end_d"])
    del frame
    forward = FC.fold_specs(cfg, ["FORWARD"])[0]
    d_min = (forward.train_end_d - window + 1) if window else None
    frame = F.load_features(cfg, d_min=d_min, d_max=forward.val_end_d)
    deploy_model = FC.train_lgbm(
        cfg, frame, forward, selected, use_early_stopping=False,
        num_boost_round=holdout_model.best_iteration, logger=log)
    forward_preds = FC.predict_window(deploy_model, frame, forward)
    if not quantiles.empty:
        forward_preds = FC.apply_residual_quantiles(cfg, forward_preds, quantiles, meta)
    _write_forecasts(cfg, forward_preds, "lgbm", selected, "FORWARD")

    # ------------------------------------------------------- explainability
    log.info("[6/6] feature importance (LightGBM native exact TreeSHAP)")
    booster = deploy_model.booster
    importance = pd.DataFrame({
        "feature": deploy_model.features,
        "gain": booster.feature_importance("gain"),
        "split": booster.feature_importance("split"),
    })
    sample_days = int(cfg["explainability"]["contrib_sample_days"])
    sample = frame[(frame["d"] >= forward.val_start_d) & (frame["d"] <= forward.val_end_d)]
    if not cfg.is_dev:
        keep_series = (
            meta.groupby(["store_id", "cat_id"], observed=True)["item_id"]
            .apply(lambda s: pd.Series(sorted(s.unique())[
                : max(1, int(cfg["explainability"]["contrib_sample_series_full"])
                      // max(1, meta.groupby(["store_id", "cat_id"]).ngroups))]))
            .reset_index(drop=True).tolist())
        sample = sample[sample["item_id"].astype(str).isin(set(keep_series))]
    sample = sample.head(200_000)
    contrib = booster.predict(sample[deploy_model.features], pred_contrib=True)
    mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
    importance["mean_abs_contrib"] = mean_abs
    importance["family"] = importance["feature"].map(F.feature_family_map())
    importance = importance.sort_values("gain", ascending=False).reset_index(drop=True)
    importance["rank"] = importance.index + 1
    importance.to_parquet(cfg.artifacts_dir / "feature_importance.parquet", index=False)
    log.info("    contributions computed on %s real rows; top feature: %s",
             f"{len(sample):,}", importance.iloc[0]["feature"])

    # ------------------------------------------------------------ artifacts
    if all_metrics:
        metrics_frame = pd.concat(all_metrics, ignore_index=True)
        metrics_frame["horizon"] = metrics_frame["horizon"].astype("Int16")
        metrics_frame.to_parquet(cfg.artifacts_dir / "metrics.parquet", index=False)
        log.info("    metrics.parquet: %s rows", f"{len(metrics_frame):,}")

    if shock_residuals:
        pd.concat(shock_residuals, ignore_index=True).to_parquet(
            cfg.artifacts_dir / "shock_residuals.parquet", index=False)

    booster.save_model(str(cfg.models_dir / "deploy_model.txt"),
                       num_iteration=deploy_model.best_iteration)
    shock_holdout_model.booster.save_model(
        str(cfg.models_dir / "shock_model.txt"),
        num_iteration=shock_holdout_model.best_iteration)

    (cfg.artifacts_dir / "fold_definitions.json").write_text(
        json.dumps({name: spec for name, spec in cfg.folds.items()}, indent=2),
        encoding="utf-8")

    calendar = D.read_processed(cfg, "calendar")
    day_to_date = dict(zip(calendar["d"], calendar["date"]))
    metadata = {
        "model_name": "DemandShock global direct LightGBM",
        "model_version": cfg["project"]["version"],
        "mode": cfg.mode,
        "config_hash": cfg.hash(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": ("direct multi-horizon; every feature frozen at the forecast "
                     "origin (target date minus 28 days)"),
        "selected_config": selected,
        "selected_config_label": F.ABLATION_LABELS.get(selected, ""),
        "selection": selection_detail,
        "shock_config": shock_config,
        "reduced_ablation_run": reduced_run,
        "objective": cfg["lgbm"]["objective"],
        "lgbm_params": {k: v for k, v in deploy_model.params.items()},
        "n_features": len(deploy_model.features),
        "features": deploy_model.features,
        "feature_families": {k: v for k, v in F.feature_family_map().items()},
        "categorical_features": deploy_model.categorical,
        "train_window_days": cfg.train_window_days,
        "training_rows_deploy": deploy_model.train_rows,
        "best_iteration_holdout": holdout_model.best_iteration,
        "best_iteration_deploy": deploy_model.best_iteration,
        "folds": {
            name: {
                **spec,
                "val_start_date": str(day_to_date[spec["val_start_d"]].date()),
                "val_end_date": str(day_to_date[spec["val_end_d"]].date()),
            } for name, spec in cfg.folds.items()
        },
        "series_count": int(len(meta)),
        "interval_coverage_holdout": coverage,
        "shap_available": _shap_available(),
        "explainability": ("LightGBM Booster.predict(pred_contrib=True) - exact "
                           "TreeSHAP values computed natively, no shap package required"),
        "library_versions": _library_versions(),
        "train_seconds_total": round(time.time() - started, 1),
    }
    (cfg.artifacts_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    log.info("-" * 68)
    log.info("Training complete in %.1f min | selected=%s | artifacts -> %s",
             (time.time() - started) / 60, selected, cfg.artifacts_dir)
    return 0


def _shap_available() -> bool:
    try:
        import shap  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _library_versions() -> dict[str, str]:
    import sys

    import numpy
    import pandas
    import pyarrow
    import scipy
    import sklearn

    return {
        "python": sys.version.split()[0],
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "lightgbm": lgb.__version__,
        "pyarrow": pyarrow.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


if __name__ == "__main__":
    raise SystemExit(main())
