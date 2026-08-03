"""Module 4 - Inventory and business impact."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shared as sh
import streamlit as st

from demandshock import inventory as INV  # noqa: E402

sh.page_setup("Inventory & business impact", ":material/inventory_2:")
ctx = sh.sidebar_context()
sh.header("INVENTORY &amp; BUSINESS IMPACT",
          "What does the forecast imply for stock, and what is at stake in dollars?")

sh.require_page("inventory_base", "model_metadata")
cfg = ctx["cfg"]
metadata = ctx["metadata"]
holdout = metadata.get("folds", {}).get("HOLDOUT", {})

base = sh.apply_filters(sh.load("inventory_base"), ctx)
if base.empty:
    sh.empty_filters()
    st.stop()

st.caption(
    f"Planning view as of {holdout.get('val_start_date', '?')}: the 28-day window "
    f"{holdout.get('val_start_date', '?')} to {holdout.get('val_end_date', '?')} was "
    "held out of training, so forecast, actual demand and dollar exposure are all "
    "measurable against each other.")

# ------------------------------------------------------------- real figures
st.subheader("Measured from real data")
total_forecast = float(base["total_forecast_28"].sum())
total_actual = float(base["total_actual_28"].sum())
under = float(base["under_exposure_usd"].sum())
over = float(base["over_exposure_usd"].sum())
revenue = float(base["actual_revenue_usd"].sum())

with st.container(horizontal=True):
    st.metric("Forecast demand (28d)", sh.fmt_units(total_forecast), border=True,
              help="Sum of the model's daily forecasts across the selection.")
    st.metric("Actual demand (28d)", sh.fmt_units(total_actual), border=True)
    st.metric("Revenue represented", sh.fmt_money(revenue), border=True,
              help="Actual units sold valued at the real M5 sell price for that week.")
with st.container(horizontal=True):
    st.metric("Under-forecast exposure", sh.fmt_money(under), border=True,
              help="Days where demand exceeded the forecast, valued at real prices. "
                   "A proxy for potential missed sales - not measured lost revenue, "
                   "because M5 records sales rather than unmet demand.")
    st.metric("Over-forecast exposure", sh.fmt_money(over), border=True,
              help="Days where the forecast exceeded demand, valued at real prices. "
                   "A proxy for the working capital tied up by over-planning.")
    st.metric("Total exposure", sh.fmt_money(under + over), border=True)

sh.note(
    "<b>Estimated revenue exposure</b> dollarises forecast error using real M5 sell "
    "prices. It is not measured lost revenue: M5 records units sold, so demand that "
    "was never met is unobservable in the source data.")

with st.container(border=True):
    st.markdown("**Exposure concentration**")
    top = base.nlargest(15, "exposure_total_usd").copy()
    top["label"] = top["item_id"].astype(str) + " @ " + top["store_id"].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top["label"], y=top["under_exposure_usd"],
                         name="Under-forecast", marker_color=sh.RED))
    fig.add_trace(go.Bar(x=top["label"], y=top["over_exposure_usd"],
                         name="Over-forecast", marker_color=sh.INDIGO))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Estimated exposure (USD)")
    share = (top["exposure_total_usd"].sum() / base["exposure_total_usd"].sum()
             if base["exposure_total_usd"].sum() else 0)
    st.plotly_chart(sh.style_fig(fig, 320), width="stretch")
    st.caption(f"These 15 item-store pairs carry {share:.0%} of the total exposure "
               f"across {len(base):,} series - which is where planner attention pays off.")

# ------------------------------------------------------- planning calculator
st.subheader("Planning parameters")
st.warning(
    "M5 contains no inventory records: no on-hand stock, no lead times, no service "
    "levels and no costs. Everything below is computed from operating assumptions "
    "**you** supply, and is labelled accordingly. Nothing is pre-filled as fact.",
    icon=":material/edit_note:")

eligible = base[base["eligible"]]
if eligible.empty:
    st.info("No series in this selection has enough price and residual history for "
            "planning arithmetic.", icon=":material/info:")
    st.stop()

chosen = sh.series_picker(eligible, key="inv_series")
if chosen is None:
    st.stop()
item_id, store_id = chosen
row = eligible[(eligible["item_id"].astype(str) == item_id)
               & (eligible["store_id"].astype(str) == store_id)].iloc[0]

with st.container(horizontal=True):
    st.metric("Forecast demand (28d)", f"{row['total_forecast_28']:.1f} units",
              border=True)
    st.metric("Mean daily forecast", f"{row['mean_daily_forecast']:.2f} units",
              border=True)
    st.metric("Daily forecast error (sigma)", f"{row['sigma_daily_resid']:.2f} units",
              border=True,
              help="Standard deviation of this series' own out-of-sample daily "
                   "forecast errors across the backtest folds. Measured, not assumed.")
    st.metric("Latest sell price", f"${row['latest_sell_price']:.2f}", border=True)

enabled = st.toggle("Enter my operating assumptions", value=False)
if not enabled:
    with st.container(border=True):
        st.markdown("**How the planning figures are calculated**")
        st.markdown(
            f"- Expected lead-time demand = sum of the daily forecasts over the lead time\n"
            f"- `{INV.SAFETY_STOCK_FORMULA}`, where *z* comes from the normal "
            f"distribution at your service level\n"
            f"- `{INV.ROP_FORMULA}`\n"
            f"- Days of cover = on-hand units divided by mean daily forecast")
        for assumption in INV.ASSUMPTIONS:
            st.caption(f"- {assumption}")
    st.info("Turn on the toggle above and enter a lead time to compute safety stock, "
            "reorder point and days of cover for this item.",
            icon=":material/toggle_off:")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1:
    lead_time = st.number_input(
        "Lead time (days)", min_value=1, max_value=int(cfg["inventory"]["max_lead_time_days"]),
        value=None, placeholder="e.g. 7",
        help="How long replenishment takes. Capped at the 28-day forecast horizon.")
with c2:
    service_level = st.selectbox(
        "Service level", cfg["inventory"]["service_levels"], index=None,
        placeholder="e.g. 0.95", format_func=lambda v: f"{v:.1%}",
        help="Probability of not stocking out during the lead time. This is a "
             "business choice, not a value derived from the data.")
with c3:
    on_hand = st.number_input("On-hand units (optional)", min_value=0.0,
                              value=None, placeholder="e.g. 40")
with c4:
    unit_cost = st.number_input("Unit cost (optional)", min_value=0.0,
                                value=None, placeholder="e.g. 1.80")

if lead_time is None or service_level is None:
    st.info("Enter a lead time and choose a service level to compute planning "
            "parameters.", icon=":material/pending:")
    st.stop()

try:
    forecasts = sh.load_forecasts(model="lgbm", config=metadata["selected_config"],
                                  fold="HOLDOUT")
except FileNotFoundError:
    forecasts = pd.DataFrame()
path_frame = INV.daily_forecast_path(forecasts, item_id, store_id)
if path_frame.empty:
    st.error("No forecast path is stored for this series.", icon=":material/error:")
    st.stop()

path = path_frame["y_pred"].to_numpy()
result = INV.plan(cfg, path, float(row["sigma_daily_resid"]), int(lead_time),
                  float(service_level), on_hand_units=on_hand, unit_cost=unit_cost,
                  unit_price=float(row["latest_sell_price"]))

if result["lead_time_clamped"]:
    st.warning(f"Lead time was capped at {result['lead_time_days_used']} days - the "
               f"model forecasts 28 days ahead, so lead-time demand cannot be "
               f"computed beyond that.", icon=":material/warning:")

st.markdown(f"{sh.user_input_badge()} &nbsp; computed from your inputs - these values "
            f"are not present in the M5 data", unsafe_allow_html=True)
with st.container(horizontal=True):
    st.metric("Expected lead-time demand", f"{result['lead_time_demand']:.1f} units",
              border=True)
    st.metric("Safety stock", f"{result['safety_stock']:.1f} units", border=True,
              help=f"{INV.SAFETY_STOCK_FORMULA}; z = {result['z']:.3f} at "
                   f"{service_level:.1%} service level.")
    st.metric("Reorder point", f"{result['reorder_point']:.1f} units", border=True,
              help=INV.ROP_FORMULA)
with st.container(horizontal=True):
    st.metric("Days of cover",
              f"{result['days_of_cover']:.1f}" if result["days_of_cover"] is not None else "-",
              border=True, help="Requires on-hand units.")
    st.metric("Projected stockout",
              f"day {result['projected_stockout_day']}"
              if result["projected_stockout_day"] else
              ("beyond 28 days" if on_hand is not None else "-"),
              border=True)
    st.metric("Uncovered demand (28d)",
              f"{result['uncovered_units']:.1f} units"
              if result["uncovered_units"] is not None else "-", border=True,
              help="Forecast demand across the horizon that the entered on-hand "
                   "position would not cover.")

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown("**Cumulative forecast demand versus your position**")
        day_map = sh.calendar_map()
        cumulative = np.cumsum(path)
        dates = path_frame["d"].map(day_map)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=cumulative, name="Cumulative forecast",
                                 mode="lines", line=dict(color=sh.CYAN, width=2.4),
                                 line_shape="hv"))
        fig.add_hline(y=result["reorder_point"], line=dict(color=sh.AMBER, dash="dot"),
                      annotation_text="Reorder point", annotation_position="top left")
        if on_hand is not None:
            fig.add_hline(y=float(on_hand), line=dict(color=sh.GREEN, dash="dash"),
                          annotation_text="On hand", annotation_position="bottom left")
            if result["projected_stockout_day"]:
                stockout_date = dates.iloc[result["projected_stockout_day"] - 1]
                fig.add_vline(x=stockout_date, line=dict(color=sh.RED))
        fig.update_yaxes(title="Cumulative units")
        st.plotly_chart(sh.style_fig(fig, 300), width="stretch")

with right:
    with st.container(border=True):
        st.markdown("**Safety stock as a function of service level**")
        curve = INV.safety_stock_curve(float(row["sigma_daily_resid"]),
                                       result["lead_time_days_used"],
                                       cfg["inventory"]["service_levels"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["service_level"], y=curve["safety_stock"],
                                 mode="lines+markers", line=dict(color=sh.TEAL, width=2)))
        fig.add_vline(x=float(service_level), line=dict(color=sh.AMBER, dash="dot"))
        fig.update_xaxes(title="Service level", tickformat=".0%")
        fig.update_yaxes(title="Safety stock (units)")
        st.plotly_chart(sh.style_fig(fig, 300), width="stretch")
        st.caption("Higher service levels cost disproportionately more stock. The "
                   "curve is the cost of the choice you make above.")

with st.expander("Assumptions behind these numbers", icon=":material/rule:"):
    st.markdown(
        f"- Independent daily errors: `sigma_lead_time = {result['sigma_daily']:.2f} "
        f"x sqrt({result['lead_time_days_used']}) = {result['sigma_lead_time']:.2f}`\n"
        f"- Conservative bound if errors were perfectly correlated: "
        f"`{result['sigma_daily']:.2f} x {result['lead_time_days_used']} = "
        f"{result['sigma_lead_time_conservative']:.2f}`, giving a safety stock of "
        f"{result['safety_stock_conservative']:.1f} units and a reorder point of "
        f"{result['reorder_point_conservative']:.1f} units")
    for assumption in result["assumptions"]:
        st.caption(f"- {assumption}")
