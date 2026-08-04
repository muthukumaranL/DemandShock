"""Module 1 - Executive command centre."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import shared as sh
import streamlit as st

sh.page_setup("Executive command centre", ":material/dashboard:")
ctx = sh.sidebar_context()
sh.header("EXECUTIVE COMMAND CENTRE",
          "Over the last evaluated 28 days, how did demand track forecast - "
          "and where is the risk?")

sh.require_page("model_metadata", "metrics", "series_meta")
metadata = ctx["metadata"]
selected_config = metadata["selected_config"]
folds = metadata.get("folds", {})
holdout = folds.get("HOLDOUT", {})

st.caption(
    f"Evaluation window: {holdout.get('val_start_date', '?')} to "
    f"{holdout.get('val_end_date', '?')} (the 28 days held out of training entirely). "
    f"Model: LightGBM, feature set {selected_config}.")

metrics = sh.load("metrics")
overall = metrics[(metrics["fold_id"] == "HOLDOUT") & (metrics["level"] == "overall")
                  & (metrics["horizon"] == 28)]
# Kept only as a fallback for the benchmark KPI when the snaive28 forecast
# partition is unavailable; the headline KPIs are computed from filtered rows.
snaive = overall[overall["model"] == "snaive28"]

# ---------------------------------------------------------------- forecasts
try:
    forecasts = sh.load_forecasts(model="lgbm", config=selected_config, fold="HOLDOUT")
except FileNotFoundError:
    forecasts = pd.DataFrame()

meta = ctx["meta"]
if not forecasts.empty:
    forecasts = forecasts.merge(
        meta[["item_id", "store_id", "dept_id", "cat_id", "state_id"]],
        on=["item_id", "store_id"], how="left")
    forecasts = sh.apply_filters(forecasts, ctx)

if forecasts.empty:
    sh.empty_filters("No forecasts match the current filters.")
    st.stop()

total_actual = float(forecasts["y_true"].sum())
total_forecast = float(forecasts["y_pred"].sum())


def _pooled(frame: pd.DataFrame) -> tuple[float, float]:
    """Pooled WAPE and bias over whatever rows are passed in.

    Computed from the FILTERED forecast rows rather than the global metrics
    artifact, so every KPI on this page describes the same selection. With no
    filters applied these reproduce the stored overall metrics exactly.
    """
    actual = float(frame["y_true"].sum())
    if actual <= 0:
        return float("nan"), float("nan")
    error = frame["y_pred"] - frame["y_true"]
    return float(error.abs().sum() / actual), float(error.sum() / actual)


wape, bias = _pooled(forecasts)

# Benchmark on the same selection. Fall back to the global row only if the
# partition is missing, and say so in the tooltip when that happens.
snaive_scope = "current selection"
try:
    bench = sh.load_forecasts(model="snaive28", config="na", fold="HOLDOUT")
    bench = bench.merge(meta[["item_id", "store_id", "dept_id", "cat_id", "state_id"]],
                        on=["item_id", "store_id"], how="left")
    bench = sh.apply_filters(bench, ctx)
    snaive_wape = _pooled(bench)[0] if not bench.empty else float("nan")
except FileNotFoundError:
    snaive_wape = float("nan")
if pd.isna(snaive_wape):
    snaive_wape = float(snaive["wape"].iloc[0]) if not snaive.empty else float("nan")
    snaive_scope = "all series"

episodes = sh.load("shock_episodes") if sh.artifact_exists("shock_episodes") else pd.DataFrame()
if not episodes.empty:
    episodes = sh.apply_filters(episodes, ctx)
severe = int(episodes["band"].isin(["Severe", "Critical"]).sum()) if not episodes.empty else 0

inventory = sh.load("inventory_base") if sh.artifact_exists("inventory_base") else pd.DataFrame()
if not inventory.empty:
    inventory = sh.apply_filters(inventory, ctx)
exposure = (float(inventory["exposure_total_usd"].sum())
            if not inventory.empty else float("nan"))

accuracy = 100 - 100 * wape if pd.notna(wape) else float("nan")
accuracy = max(0.0, accuracy) if pd.notna(accuracy) else accuracy
improvement = ((snaive_wape - wape) / snaive_wape) if pd.notna(snaive_wape) and snaive_wape else float("nan")

severe_threshold = int(ctx["cfg"]["shock"]["bands"]["Severe"][0])

with st.container(horizontal=True):
    st.metric("Actual units (28d)", sh.fmt_units(total_actual),
              delta=f"{total_forecast - total_actual:+,.0f} vs forecast", border=True)
    st.metric("Item-day accuracy (100-WAPE)",
              f"{accuracy:.1f}%" if pd.notna(accuracy) else "-", border=True,
              help=f"100 - WAPE, clamped at zero, measured per item per store per "
                   f"day on the current selection. Raw WAPE is {wape:.3f}. "
                   "WAPE can exceed 100% on intermittent demand.")
    st.metric("Forecast bias",
              f"{bias * 100:+.1f}%" if pd.notna(bias) else "-", border=True,
              help="Positive means the model over-forecast overall; negative means "
                   "it under-forecast. Computed on the held-out 28 days for the "
                   "current selection.")
with st.container(horizontal=True):
    st.metric("Severe + critical shocks", f"{severe:,}", border=True,
              help=f"Demand-shock episodes in the evaluated window scoring "
                   f"{severe_threshold} or above.")
    st.metric("Estimated revenue exposure", sh.fmt_money(exposure), border=True,
              help="Forecast error valued at real M5 sell prices over the 28-day "
                   "window. This is exposure, not measured lost revenue - M5 records "
                   "sales, not unmet demand.")
    st.metric("Gain vs seasonal naive",
              sh.fmt_pct(improvement) if pd.notna(improvement) else "-", border=True,
              help=f"Reduction in WAPE against a seasonal-naive (lag-28) benchmark "
                   f"that forecasts from the same 28-day-old information. Benchmark "
                   f"WAPE {snaive_wape:.3f} ({snaive_scope}).")

if pd.notna(bias) and pd.notna(accuracy):
    direction = "over-forecast" if bias > 0 else "under-forecast"
    message = (
        f"**Read these two together.** Accuracy here is measured per item, per store, "
        f"per day, where demand is intermittent - most series sell a handful of units "
        f"on some days and none on others, so matching individual days exactly is "
        f"inherently hard at that granularity. Across the current selection the 28-day "
        f"**total** came within {abs(bias) * 100:.1f}% of actual demand ({direction}).")
    if pd.notna(improvement):
        message += (f" The number that matters for a build-or-buy case is the "
                    f"{improvement:.0%} reduction in error against the seasonal-naive "
                    f"benchmark.")
    st.caption(message)

# ---------------------------------------------------------------- charts
left, right = st.columns(2)

with left:
    with st.container(border=True):
        st.markdown("**Daily demand: actual vs forecast**")
        day_map = sh.calendar_map()
        daily = (forecasts.groupby("d", as_index=False)[["y_true", "y_pred"]].sum()
                 .sort_values("d"))
        daily["date"] = daily["d"].map(day_map)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["y_true"], name="Actual",
                                 mode="lines", line=dict(color=sh.CYAN, width=2.4)))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["y_pred"], name="Forecast",
                                 mode="lines",
                                 line=dict(color=sh.AMBER, width=2, dash="dot")))
        fig.update_yaxes(title="Units")
        st.plotly_chart(sh.style_fig(fig, 300), width="stretch")
        st.caption("Held-out 28 days. The model never saw these days during training.")

with right:
    with st.container(border=True):
        st.markdown("**Where the exposure sits**")
        if inventory.empty:
            sh.missing_section("inventory_base", "Revenue exposure")
        else:
            top = inventory.nlargest(10, "exposure_total_usd")
            lookup = sh.label_lookup()
            label = [
                (lookup.get((str(i), str(s)), f"{i} @ {s}") + f"<br>{i} @ {s}")
                for i, s in zip(top["item_id"], top["store_id"])
            ]
            fig = go.Figure()
            fig.add_trace(go.Bar(y=label, x=top["under_exposure_usd"],
                                 name="Under-forecast", orientation="h",
                                 marker_color=sh.RED))
            fig.add_trace(go.Bar(y=label, x=top["over_exposure_usd"],
                                 name="Over-forecast", orientation="h",
                                 marker_color=sh.INDIGO))
            fig.update_layout(barmode="stack")
            fig.update_xaxes(title="Estimated exposure (USD)")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(sh.style_fig(fig, 300), width="stretch")
            st.caption("Top 10 item-store pairs by forecast error valued at real "
                       "sell prices. Under-forecast risks lost sales; over-forecast "
                       "risks excess stock.")

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown("**Demand shocks by department and severity**")
        if episodes.empty:
            sh.missing_section("shock_episodes", "Shock episodes")
        else:
            pivot = (episodes.groupby(["dept_id", "band"], observed=True)
                     .size().reset_index(name="n"))
            fig = go.Figure()
            for band in sh.BAND_ORDER:
                block = pivot[pivot["band"] == band]
                if block.empty:
                    continue
                fig.add_trace(go.Bar(x=block["dept_id"], y=block["n"], name=band,
                                     marker_color=sh.BAND_COLORS[band]))
            fig.update_layout(barmode="stack")
            fig.update_yaxes(title="Episodes")
            st.plotly_chart(sh.style_fig(fig, 300), width="stretch")
            st.caption("Episodes across the full evaluated window, not only the holdout.")

with right:
    with st.container(border=True):
        st.markdown("**Daily forecast error through the evaluated window**")
        daily_metrics = metrics[(metrics["level"] == "day")
                                & (metrics["model"] == "lgbm")
                                & (metrics["config"] == selected_config)]
        if daily_metrics.empty:
            st.caption("Daily metric series unavailable.")
        else:
            day_map = sh.calendar_map()
            series = daily_metrics.copy()
            series["d"] = series["group_key"].astype(int)
            series["date"] = series["d"].map(day_map)
            series = series.sort_values("date")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series["date"], y=series["wape"],
                                     mode="lines", line=dict(color=sh.TEAL, width=2),
                                     name="WAPE"))
            fig.update_yaxes(title="WAPE", tickformat=".0%")
            st.plotly_chart(sh.style_fig(fig, 300), width="stretch")
            st.caption("Lower is better. Spikes mark days the model tracked demand "
                       "least well. This chart covers **all series across the full "
                       "evaluated window** (backtest folds plus the holdout) and is "
                       "not affected by the sidebar filters.")

if not episodes.empty:
    with st.container(border=True):
        st.markdown("**Most severe demand shocks needing review**")
        top = sh.attach_labels(episodes.nlargest(10, "peak_score")[
            ["item_id", "store_id", "start_date", "end_date", "n_days",
             "classification", "band", "peak_score", "fema_overlap"]])
        st.dataframe(
            top, hide_index=True, width="stretch",
            column_config={
                "Product": st.column_config.TextColumn(
                    "Product", width="medium",
                    help="Measured description: department, price band within its "
                         "category, and sales velocity. M5 anonymises product names."),
                "peak_score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=100, format="%.0f"),
                "start_date": st.column_config.DateColumn("Start"),
                "end_date": st.column_config.DateColumn("End"),
                "n_days": st.column_config.NumberColumn("Days"),
                "fema_overlap": st.column_config.CheckboxColumn("FEMA active"),
            })
        st.caption("'FEMA active' means a declaration was in force in that state "
                   "during the episode - a coincidence in time, not a cause. "
                   "Open module 3 for the full reasoning behind each score.")
