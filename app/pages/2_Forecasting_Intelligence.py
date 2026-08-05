"""Module 2 - Forecasting intelligence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shared as sh
import streamlit as st

sh.page_setup("Forecasting intelligence", ":material/query_stats:")
ctx = sh.sidebar_context()
sh.header("FORECASTING INTELLIGENCE",
          "How accurate is the forecast - for which products, stores and horizons?")

sh.require_page("model_metadata", "metrics", "series_meta")
metadata = ctx["metadata"]
selected_config = metadata["selected_config"]
metrics = sh.load("metrics")
meta = ctx["meta"]
day_map = sh.calendar_map()

folds_available = [f for f in ["F1", "F2", "F3", "HOLDOUT", "FORWARD"]
                   if f in set(metrics["fold_id"]) or f == "FORWARD"]

controls = st.container()
with controls:
    c1, c2, c3 = st.columns(3)
    with c1:
        fold = st.selectbox(
            "Evaluation window", folds_available,
            index=folds_available.index("HOLDOUT") if "HOLDOUT" in folds_available else 0,
            help="F1-F3 are rolling-origin backtest folds. HOLDOUT was opened once, "
                 "after model selection. FORWARD is the genuine forward forecast "
                 "past the end of the data, so it has no actuals.")
    with c2:
        horizon = st.selectbox("Horizon (days)", ctx["cfg"]["horizons"], index=2,
                               help="Metrics are computed over the first N days of "
                                    "the 28-day forecast path.")
    with c3:
        available_models = sorted(set(metrics["model"]))
        chosen_models = st.multiselect(
            "Models to compare", available_models,
            default=[m for m in ["lgbm", "snaive28", "naive"] if m in available_models],
            format_func=lambda m: sh.MODEL_LABELS.get(m, m))

fold_info = metadata.get("folds", {}).get(fold, {})
if fold == "FORWARD":
    st.info(
        f"Forward forecast for {fold_info.get('val_start_date', '?')} to "
        f"{fold_info.get('val_end_date', '?')}, produced by the deploy model trained "
        "through the final day of the dataset. **These days have no actuals**, so no "
        "accuracy metric can be computed for them.",
        icon=":material/trending_up:")
else:
    st.caption(f"Window {fold_info.get('val_start_date', '?')} to "
               f"{fold_info.get('val_end_date', '?')} - "
               f"{'held out of training entirely' if fold == 'HOLDOUT' else 'rolling-origin backtest fold'}.")

# ------------------------------------------------------------------ metrics
if fold != "FORWARD":
    slice_ = metrics[(metrics["fold_id"] == fold) & (metrics["level"] == "overall")
                     & (metrics["horizon"] == horizon)
                     & (metrics["model"].isin(chosen_models))]
    primary = slice_[(slice_["model"] == "lgbm") & (slice_["config"] == selected_config)]
    if primary.empty:
        primary = slice_[slice_["model"] == "lgbm"]
    if not primary.empty:
        row = primary.iloc[0]
        with st.container(horizontal=True):
            for name in ["mae", "rmse", "rmsse", "wape", "smape", "bias"]:
                digits = 3 if name != "smape" else 1
                st.metric(name.upper(), sh.fmt_metric(row[name], digits),
                          border=True, help=sh.METRIC_HELP[name])
        excluded = int(row.get("n_rmsse_excluded", 0) or 0)
        if excluded:
            st.caption(f"RMSSE excludes {excluded} series whose training history was "
                       f"completely flat - the scaling denominator is zero for them, "
                       f"so the metric is undefined rather than fudged.")

# ------------------------------------------------------------------ series view
st.subheader("Forecast versus actual")
selected_series = ctx.get("selected_series")
if selected_series is None or selected_series.empty:
    sh.empty_filters()
    st.stop()

view_col, item_col = st.columns([1, 2])
with view_col:
    view = st.radio("View", ["Aggregate of selection", "Single item"], horizontal=False)
with item_col:
    chosen = sh.series_picker(selected_series, key="fc_series",
                              disabled=(view != "Single item"))

try:
    lgbm_fc = sh.load_forecasts(model="lgbm", config=selected_config, fold=fold)
except FileNotFoundError:
    lgbm_fc = pd.DataFrame()

if lgbm_fc.empty:
    st.info(f"No LightGBM forecasts stored for {fold}.", icon=":material/info:")
    st.stop()

lgbm_fc = sh.apply_filters(lgbm_fc, ctx)
if view == "Single item" and chosen:
    item_id, store_id = chosen
    lgbm_fc = lgbm_fc[(lgbm_fc["item_id"].astype(str) == item_id)
                      & (lgbm_fc["store_id"].astype(str) == store_id)]

if lgbm_fc.empty:
    sh.empty_filters()
    st.stop()

agg_cols = {"y_pred": "sum"}
if "y_true" in lgbm_fc.columns:
    agg_cols["y_true"] = "sum"
for band in ("p10", "p90"):
    if band in lgbm_fc.columns:
        agg_cols[band] = "sum"
path = lgbm_fc.groupby("d", as_index=False).agg(agg_cols).sort_values("d")
path["date"] = path["d"].map(day_map)

# Daily item-level demand is spiky and mostly zero, so a daily chart of one slow
# mover looks like a miss even when the forecast is the best available estimate.
# Weekly and cumulative views show the same forecast at the granularity a planner
# actually orders on.
grain = st.radio(
    "Chart grain", ["Daily", "Weekly total", "Cumulative"], horizontal=True,
    help="Daily is the raw series. Weekly totals and the cumulative curve show "
         "whether the forecast is right in aggregate, which is what replenishment "
         "depends on - a forecast of 0.7 units/day cannot match a day that sells "
         "0 or 3, but can still be right across the week.")

history_days = 56
first_day = int(path["d"].min())
try:
    sales = sh.load_sales(d_min=first_day - history_days)
    sales = sales[sales["d"] < first_day]
    sales = sh.apply_filters(
        sales.merge(meta[["item_id", "store_id"]], on="item_id", how="left")
        if "store_id" not in sales.columns else sales, ctx)
    if view == "Single item" and chosen:
        sales = sales[(sales["item_id"].astype(str) == item_id)
                      & (sales["store_id"].astype(str) == store_id)]
    history = sales.groupby("d", as_index=False)["sales"].sum().sort_values("d")
    history["date"] = history["d"].map(day_map)
except FileNotFoundError:
    history = pd.DataFrame()

value_cols = [c for c in ("y_true", "y_pred", "p10", "p90") if c in path.columns]
if grain == "Weekly total":
    path = (path.assign(bucket=((path["d"] - path["d"].min()) // 7))
            .groupby("bucket", as_index=False)
            .agg({**{c: "sum" for c in value_cols}, "date": "last"}))
    if not history.empty:
        history = (history.assign(bucket=((history["d"] - history["d"].min()) // 7))
                   .groupby("bucket", as_index=False)
                   .agg({"sales": "sum", "date": "last"}))
elif grain == "Cumulative":
    path = path.sort_values("date").copy()
    for col in value_cols:
        path[col] = path[col].cumsum()
    history = pd.DataFrame()      # a cumulative history would dwarf the window

with st.container(border=True):
    fig = go.Figure()
    if not history.empty:
        fig.add_trace(go.Scatter(x=history["date"], y=history["sales"],
                                 name="Actual (history)", mode="lines",
                                 line=dict(color=sh.MUTED, width=1.5)))
    if "p10" in path.columns and "p90" in path.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([path["date"], path["date"][::-1]]),
            y=pd.concat([path["p90"], path["p10"][::-1]]),
            fill="toself", fillcolor="rgba(34,211,238,0.13)",
            line=dict(width=0), hoverinfo="skip", name="P10-P90"))
    if "y_true" in path.columns and path["y_true"].notna().any():
        fig.add_trace(go.Scatter(x=path["date"], y=path["y_true"], name="Actual",
                                 mode="lines", line=dict(color=sh.CYAN, width=2.4)))
    fig.add_trace(go.Scatter(x=path["date"], y=path["y_pred"], name="Forecast",
                             mode="lines",
                             line=dict(color=sh.AMBER, width=2, dash="dot")))
    fig.update_yaxes(title={"Daily": "Units per day",
                            "Weekly total": "Units per week",
                            "Cumulative": "Cumulative units"}[grain])
    if view == "Single item" and chosen:
        st.markdown(f"**{sh.describe_series(item_id, store_id, with_id=False)}**")
        st.caption(f"{item_id} @ {store_id} · {sh.ANONYMITY_NOTE}")
    st.plotly_chart(sh.style_fig(fig, 360), width="stretch")
    caption = "Grey is observed history before the window. "
    if "p10" in path.columns:
        coverage = metadata.get("interval_coverage_holdout", {}).get("coverage")
        caption += ("The shaded band is the P10-P90 interval from empirical "
                    "out-of-sample residuals")
        caption += (f"; measured coverage on the holdout was {coverage:.0%} "
                    f"against an 80% target. " if coverage else ". ")
    if fold == "FORWARD":
        caption += "This window has no actuals - it is a genuine forward forecast."
    st.caption(caption)

    # On an intermittent series the dotted line will never touch the spikes, and
    # that is not a defect: the forecast is an expected value, and the band is the
    # range being predicted. Say so where the reader would otherwise conclude the
    # model is simply wrong.
    if grain == "Daily" and view == "Single item" and chosen \
            and path["y_pred"].mean() < 5:
        inside = None
        if {"p10", "p90", "y_true"}.issubset(path.columns) and path["y_true"].notna().any():
            inside = float(((path["y_true"] >= path["p10"])
                            & (path["y_true"] <= path["p90"])).mean())
        st.info(
            "**Reading a low-volume item.** The forecast line is an *expected value* "
            f"of about {path['y_pred'].mean():.1f} units a day, not a prediction that "
            "a given day will sell exactly that. A product that sells a few units a "
            "week has no pattern saying *which* day the sale lands, so the dotted "
            "line will never sit on the spikes — the shaded band is the range being "
            "predicted."
            + (f" Here **{inside:.0%} of actual days fell inside the band**."
               if inside is not None else "")
            + " Switch the grain to **Weekly total** or **Cumulative** to see whether "
              "the forecast is right at the level you would actually order on.",
            icon=":material/insights:")

# ------------------------------------------------------------------ comparisons
if fold != "FORWARD":
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**Accuracy by horizon and model**")
            block = metrics[(metrics["fold_id"] == fold) & (metrics["level"] == "overall")
                            & (metrics["model"].isin(chosen_models))]
            block = block[(block["model"] != "lgbm")
                          | (block["config"] == selected_config)]
            if block.empty:
                st.caption("No comparable metrics for this selection.")
            else:
                fig = go.Figure()
                for model in chosen_models:
                    sub = block[block["model"] == model].sort_values("horizon")
                    if sub.empty:
                        continue
                    fig.add_trace(go.Bar(x=sub["horizon"].astype(str), y=sub["wape"],
                                         name=sh.MODEL_LABELS.get(model, model)))
                fig.update_yaxes(title="WAPE", tickformat=".0%")
                fig.update_xaxes(title="Horizon (days)", type="category")
                st.plotly_chart(sh.style_fig(fig, 300), width="stretch")
                st.caption("Seasonal naive (lag 28) shares the model's information "
                           "set exactly, which makes it the fair benchmark.")

    with right:
        with st.container(border=True):
            st.markdown("**Stability across backtest folds**")
            block = metrics[(metrics["level"] == "overall")
                            & (metrics["horizon"] == horizon)
                            & (metrics["model"].isin(chosen_models))]
            block = block[(block["model"] != "lgbm")
                          | (block["config"] == selected_config)]
            order = ["F1", "F2", "F3", "HOLDOUT"]
            block = block[block["fold_id"].isin(order)]
            if block.empty:
                st.caption("No fold comparison available.")
            else:
                fig = go.Figure()
                for model in chosen_models:
                    sub = block[block["model"] == model]
                    sub = sub.set_index("fold_id").reindex(order).reset_index()
                    fig.add_trace(go.Scatter(
                        x=sub["fold_id"], y=sub["wape"], mode="lines+markers",
                        name=sh.MODEL_LABELS.get(model, model)))
                fig.update_yaxes(title="WAPE", tickformat=".0%")
                st.plotly_chart(sh.style_fig(fig, 300), width="stretch")
                st.caption("Fold-to-fold movement shows how much of any difference "
                           "between models is real rather than window-specific noise.")

    with st.container(border=True):
        st.markdown("**Where the model performs worst**")
        level = st.segmented_control("Break down by",
                                     ["store", "category", "department", "state"],
                                     default="department")
        block = metrics[(metrics["fold_id"] == fold) & (metrics["level"] == level)
                        & (metrics["horizon"] == horizon)
                        & (metrics["model"] == "lgbm")
                        & (metrics["config"] == selected_config)]
        if block.empty:
            st.caption("No breakdown available for this selection.")
        else:
            table = block[["group_key", "wape", "rmsse", "bias", "mae",
                           "n_series", "n_obs"]].sort_values("wape", ascending=False)
            st.dataframe(
                table, hide_index=True, width="stretch",
                column_config={
                    "group_key": st.column_config.TextColumn(level.title()),
                    "wape": st.column_config.NumberColumn("WAPE", format="%.3f"),
                    "rmsse": st.column_config.NumberColumn("RMSSE", format="%.3f"),
                    "bias": st.column_config.NumberColumn("Bias", format="%+.3f"),
                    "mae": st.column_config.NumberColumn("MAE", format="%.3f"),
                    "n_series": st.column_config.NumberColumn("Series"),
                    "n_obs": st.column_config.NumberColumn("Item-days"),
                })
            st.caption("Sorted worst first. Bad segments are shown, not hidden.")

    with st.container(border=True):
        st.markdown("**Residual distribution**")
        if "y_true" in lgbm_fc.columns and lgbm_fc["y_true"].notna().any():
            residual = lgbm_fc["y_true"] - lgbm_fc["y_pred"]
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=residual, nbinsx=60, marker_color=sh.TEAL))
            fig.add_vline(x=0, line=dict(color=sh.MUTED, dash="dash"))
            fig.update_xaxes(title="Actual minus forecast (units)")
            fig.update_yaxes(title="Item-days")
            st.plotly_chart(sh.style_fig(fig, 280), width="stretch")
            st.caption(f"Mean residual {residual.mean():+.3f} units, median "
                       f"{residual.median():+.3f}. A long right tail means occasional "
                       "large under-forecasts, which is typical of intermittent demand.")
        else:
            st.caption("Residuals require actuals; this window has none.")
