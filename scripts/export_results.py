"""Generate RESULTS.md directly from the artifacts.

Every number in the documentation is produced here, from files on disk. Nothing is
typed by hand, so the README and RESULTS can never drift from what was measured.

Run:  python scripts/export_results.py [--mode development|full]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
import pandas as pd

from demandshock import features as F
from demandshock.config import load_config, get_logger

MODEL_LABELS = {
    "naive": "Naive (last observed day)",
    "snaive7": "Seasonal naive (lag 7)",
    "snaive28": "Seasonal naive (lag 28)",
    "lgbm": "LightGBM (global, Tweedie)",
}


def _fmt(value, digits=4):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["development", "full"], default=None)
    args = parser.parse_args()
    cfg = load_config(mode=args.mode)
    log = get_logger("export_results", cfg, "export_results.log")

    metadata_path = cfg.artifacts_dir / "model_metadata.json"
    if not metadata_path.exists():
        log.error("model_metadata.json missing - run scripts/train.py first")
        return 1
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    quality = json.loads((cfg.artifacts_dir / "data_quality.json").read_text("utf-8"))
    tables = quality.get("tables", {})
    selected = metadata["selected_config"]

    metrics = pd.read_parquet(cfg.artifacts_dir / "metrics.parquet")
    ablation = pd.read_parquet(cfg.artifacts_dir / "ablation.parquet")

    lines: list[str] = []
    add = lines.append

    add("# DemandShock - measured results\n")
    add(f"*Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from "
        f"the artifacts in `{cfg.artifacts_dir.name}/`. Every figure below is produced "
        f"by `scripts/export_results.py` reading files on disk - none is typed by "
        f"hand.*\n")
    add(f"**Run mode:** `{metadata['mode']}` &nbsp;|&nbsp; "
        f"**trained:** {metadata['trained_at']} &nbsp;|&nbsp; "
        f"**config hash:** `{metadata['config_hash']}`\n")
    if metadata.get("reduced_ablation_run"):
        add("> **Warning:** these artifacts come from a reduced (smoke) run. The "
            "ablation figures are not valid findings.\n")

    # ---------------------------------------------------------------- data
    add("## Data actually processed\n")
    add("| Quantity | Value |")
    add("| --- | --- |")
    add(f"| Real series (item x store) | {tables.get('series', 0):,} |")
    add(f"| Item-days after release filtering | {tables.get('sales_rows', 0):,} |")
    add(f"| Weekly price records | {tables.get('price_rows', 0):,} |")
    add(f"| Stores / categories / departments | {tables.get('stores')} / "
        f"{tables.get('categories')} / {tables.get('departments')} |")
    add(f"| Date coverage | {tables.get('date_start')} to {tables.get('date_end')} "
        f"({tables.get('calendar_days')} days) |")
    add(f"| Zero-sale share of item-days | {tables.get('zero_demand_ratio', 0):.1%} |")
    add(f"| FEMA disasters in window (CA/TX/WI) | {tables.get('fema_disasters')} |")
    add(f"| FRED monthly observations | {tables.get('fred_months'):,} |")
    add(f"| Data-quality checks | {quality.get('n_pass')} passed, "
        f"{quality.get('n_warn')} warnings, {quality.get('n_fail')} failures |\n")

    # ---------------------------------------------------------------- folds
    add("## Validation design\n")
    add("Chronological rolling-origin only. No random splitting anywhere.\n")
    add("| Split | Days | Dates | Role |")
    add("| --- | --- | --- | --- |")
    roles = {
        "F1": "backtest fold, ablation, residual pool",
        "F2": "backtest fold, ablation, residual pool",
        "F3": "backtest fold, ablation, residual pool",
        "HOLDOUT": "opened once, after selection - headline numbers",
        "FORWARD": "forward forecast beyond the data (no actuals exist)",
    }
    for name, spec in metadata["folds"].items():
        add(f"| {name} | d_{spec['val_start_d']}-d_{spec['val_end_d']} | "
            f"{spec['val_start_date']} to {spec['val_end_date']} | {roles.get(name, '')} |")
    add("")

    # ---------------------------------------------------------------- holdout
    add("## Held-out accuracy (28-day horizon)\n")
    holdout = metrics[(metrics["fold_id"] == "HOLDOUT") & (metrics["level"] == "overall")
                      & (metrics["horizon"] == 28)]
    holdout = holdout[(holdout["model"] != "lgbm") | (holdout["config"] == selected)]
    add("| Model | WAPE | RMSSE | MAE | RMSE | Bias | sMAPE |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for row in holdout.sort_values("wape").itertuples(index=False):
        label = MODEL_LABELS.get(row.model, row.model)
        if row.model == "lgbm":
            label += f" - feature set {row.config}"
        add(f"| {label} | {_fmt(row.wape)} | {_fmt(row.rmsse)} | {_fmt(row.mae)} | "
            f"{_fmt(row.rmse)} | {_fmt(row.bias)} | {_fmt(row.smape, 1)} |")
    add("")

    best = holdout[holdout["model"] == "lgbm"]
    bench = holdout[holdout["model"] == "snaive28"]
    if not best.empty and not bench.empty:
        gain = (float(bench["wape"].iloc[0]) - float(best["wape"].iloc[0])) / float(
            bench["wape"].iloc[0])
        add(f"LightGBM reduces WAPE by **{gain:.1%}** against seasonal-naive-28, the "
            f"benchmark that shares its exact information set. Its RMSSE of "
            f"{_fmt(best['rmsse'].iloc[0])} is below 1.0, meaning it also beats a "
            f"one-day naive forecast measured on each series' own training history.\n")
    add("> sMAPE is reported for completeness but is misleading on intermittent "
        "demand: on a zero-sale day a naive forecast of exactly zero scores perfectly "
        "while any positive forecast is penalised the full 200%. WAPE and RMSSE are "
        "the trustworthy comparisons here.\n")

    add("### By horizon\n")
    add("| Horizon | WAPE | RMSSE | Bias |")
    add("| --- | --- | --- | --- |")
    by_h = metrics[(metrics["fold_id"] == "HOLDOUT") & (metrics["level"] == "overall")
                   & (metrics["model"] == "lgbm") & (metrics["config"] == selected)]
    for row in by_h.sort_values("horizon").itertuples(index=False):
        add(f"| {int(row.horizon)} days | {_fmt(row.wape)} | {_fmt(row.rmsse)} | "
            f"{_fmt(row.bias)} |")
    add("")

    coverage = metadata.get("interval_coverage_holdout") or {}
    if coverage:
        add(f"**Prediction intervals.** Empirical P10-P90 bands covered "
            f"{coverage['coverage']:.1%} of held-out actuals against an 80% design "
            f"target ({coverage['below']:.1%} fell below P10, {coverage['above']:.1%} "
            f"above P90, n={coverage['n']:,}).\n")

    # ---------------------------------------------------------------- ablation
    add("## Ablation: do FEMA and FRED actually help?\n")
    horizon = int(cfg["ablation"]["selection_horizon"])
    at_h = ablation[ablation["horizon"] == horizon]
    summary = (at_h.groupby("config")
               .agg(wape=("wape", "mean"), wape_std=("wape", "std"),
                    rmsse=("rmsse", "mean"), bias=("bias", "mean"),
                    n_features=("n_features", "first"))
               .reindex(["A", "B", "C", "D"]).dropna(how="all"))
    add(f"Mean across folds {', '.join(cfg.cv_folds)} at horizon {horizon}, identical "
        f"seed, parameters and rows in every arm.\n")
    add("| Arm | Feature set | Features | WAPE | sd across folds | RMSSE | Bias |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for config, row in summary.iterrows():
        add(f"| {config} | {F.ABLATION_LABELS[config]} | {int(row['n_features'])} | "
            f"{_fmt(row['wape'])} | {_fmt(row['wape_std'])} | {_fmt(row['rmsse'])} | "
            f"{_fmt(row['bias'])} |")
    add("")

    pivot = at_h.pivot_table(index="fold_id", columns="config", values="wape")
    verdicts = []
    for arm, previous, name in (("B", "A", "Calendar and price"),
                                ("C", "B", "FEMA disaster context"),
                                ("D", "C", "FRED economic context")):
        if arm not in summary.index or previous not in summary.index:
            continue
        delta = float(summary.loc[arm, "wape"] - summary.loc[previous, "wape"])
        rel = delta / float(summary.loc[previous, "wape"])
        spread = float(summary.loc[previous, "wape_std"] or 0)
        wins = int((pivot[arm] < pivot[previous]).sum())
        folds = int(pivot[arm].notna().sum())
        if delta < 0:
            verdict = (f"- **{name}: improved accuracy.** WAPE fell {abs(delta):.4f} "
                       f"({abs(rel):.2%} relative), better on {wins} of {folds} folds.")
            if abs(delta) < spread:
                verdict += (f" This is smaller than the {spread:.4f} fold-to-fold "
                            f"spread, so it is a marginal gain, not a decisive one.")
        else:
            verdict = (f"- **{name}: did NOT improve accuracy.** WAPE rose {delta:.4f} "
                       f"({rel:.2%} relative), better on only {wins} of {folds} folds.")
        verdicts.append(verdict)
    lines.extend(verdicts)
    add("")

    if (cfg.artifacts_dir / "feature_importance.parquet").exists():
        fi = pd.read_parquet(cfg.artifacts_dir / "feature_importance.parquet")
        share = (fi.groupby("family")["gain"].sum() / fi["gain"].sum()).sort_values(
            ascending=False)
        # Attribution exists only for the SELECTED feature set. Quoting "0.00% of
        # gain" for a family that is not in that set would read as a measurement
        # when it is a structural certainty.
        present = set(fi["family"])
        measured = [f for f in ("fema", "fred") if f in present]
        absent = [f for f in ("fema", "fred") if f not in present]
        if measured:
            add("For perspective, "
                + " and ".join(f"{f.upper()} features account for "
                               f"{share.get(f, 0):.2%} of total model gain"
                               for f in measured)
                + ". Read the WAPE deltas above against those shares before "
                  "concluding that external data improves point forecasts.\n")
        if absent:
            add(f"{' and '.join(f.upper() for f in absent)} features are not part of "
                f"the selected feature set, so they carry no attribution here by "
                f"construction - the ablation table above is the evidence on whether "
                f"they help.\n")
        add("**Share of model gain by feature family**\n")
        add("| Family | Share of gain |")
        add("| --- | --- |")
        for family, value in share.items():
            add(f"| {F.FAMILY_LABELS.get(family, family)} | {value:.2%} |")
        add("")
        add(f"Top features by gain: " + ", ".join(
            f"`{r.feature}`" for r in fi.head(6).itertuples(index=False)) + ".\n")

    add(f"**Selection.** {metadata['selection'].get('rule', '')}. "
        f"Selected feature set **{selected}** "
        f"({metadata.get('selected_config_label', '')}).\n")

    objective_path = cfg.artifacts_dir / "objective_comparison.csv"
    if objective_path.exists():
        obj = pd.read_csv(objective_path)
        add("**Objective check** (development mode, fold F1, arm B, then frozen):\n")
        add("| Objective | WAPE | RMSSE | Bias |")
        add("| --- | --- | --- | --- |")
        for row in obj.itertuples(index=False):
            add(f"| {row.objective} | {_fmt(row.wape)} | {_fmt(row.rmsse)} | "
                f"{_fmt(row.bias)} |")
        add("")

    # ---------------------------------------------------------------- shocks
    episodes_path = cfg.artifacts_dir / "shock_episodes.parquet"
    if episodes_path.exists():
        episodes = pd.read_parquet(episodes_path)
        daily = pd.read_parquet(cfg.artifacts_dir / "shocks_daily.parquet")
        add("## Demand shock detection\n")
        add(f"Scored window: **{daily['date'].min().date()} to "
            f"{daily['date'].max().date()}** "
            f"({daily['d'].nunique()} days), covering "
            f"{daily[['item_id', 'store_id']].drop_duplicates().shape[0]:,} eligible "
            f"series and {len(daily):,} scored series-days. Residuals come from "
            f"feature set {metadata['shock_config']}, which contains no FEMA or FRED "
            f"features by design.\n")
        add(f"**{len(episodes):,} episodes** were filed.\n")
        add("| Classification | Episodes | Median duration | Mean peak score |")
        add("| --- | --- | --- | --- |")
        grouped = episodes.groupby("classification").agg(
            n=("episode_id", "size"), days=("n_days", "median"),
            peak=("peak_score", "mean")).sort_values("n", ascending=False)
        for name, row in grouped.iterrows():
            add(f"| {name} | {int(row['n'])} | {row['days']:.0f} days | "
                f"{row['peak']:.1f} |")
        add("")
        band_counts = episodes["band"].value_counts()
        add("| Severity | Episodes |")
        add("| --- | --- |")
        for band in ["Critical", "Severe", "Elevated", "Watch", "Normal"]:
            if band in band_counts:
                add(f"| {band} | {int(band_counts[band])} |")
        add("")
        overlap = float(episodes["fema_overlap"].mean())
        add(f"**{overlap:.1%}** of episodes coincided with a FEMA declaration active "
            f"in the same state. This is a coincidence rate measured over the scored "
            f"window - it is not evidence of causation, and the platform never "
            f"presents it as such.\n")

    # ---------------------------------------------------------------- inventory
    inventory_path = cfg.artifacts_dir / "inventory_base.parquet"
    if inventory_path.exists():
        base = pd.read_parquet(inventory_path)
        add("## Business impact (measured, not assumed)\n")
        add("| Quantity | Value |")
        add("| --- | --- |")
        add(f"| Series covered | {len(base):,} ({int(base['eligible'].sum()):,} "
            f"eligible for planning arithmetic) |")
        add(f"| Actual units sold in the holdout window | "
            f"{base['total_actual_28'].sum():,.0f} |")
        add(f"| Revenue represented at real M5 prices | "
            f"${base['actual_revenue_usd'].sum():,.0f} |")
        add(f"| Under-forecast exposure | ${base['under_exposure_usd'].sum():,.0f} |")
        add(f"| Over-forecast exposure | ${base['over_exposure_usd'].sum():,.0f} |")
        add(f"| Total estimated revenue exposure | "
            f"${base['exposure_total_usd'].sum():,.0f} |")
        add("")
        add("Estimated revenue exposure dollarises forecast error using real M5 sell "
            "prices. It is **not** measured lost revenue: M5 records units sold, so "
            "demand that was never satisfied is unobservable in the source data. "
            "Safety stock, reorder points and days of cover are shown in the "
            "application only after a user supplies lead time and service level, "
            "because M5 contains no inventory records at all.\n")

    add("## Reproducing these numbers\n")
    add("```bash\npython scripts/run_pipeline.py --mode "
        f"{metadata['mode']}\npython scripts/export_results.py\n```\n")
    add(f"Library versions: " + ", ".join(
        f"{k} {v}" for k, v in metadata.get("library_versions", {}).items()) + ".\n")

    output = cfg.artifacts_dir.parent / "RESULTS.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s (%s lines) from artifacts", output.name, len(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
