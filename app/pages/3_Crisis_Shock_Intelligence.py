"""Module 3 - Crisis and demand shock intelligence (the DemandShock differentiator)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shared as sh
import streamlit as st

from demandshock import shock as S  # noqa: E402  (shared.py puts src/ on sys.path)

sh.page_setup("Crisis & demand shock intelligence", ":material/crisis_alert:")
ctx = sh.sidebar_context()
sh.header("CRISIS &amp; DEMAND SHOCK INTELLIGENCE",
          "Where did demand break from expectation, how severe was it, and what "
          "real-world events coincided?")

sh.require_page("shock_episodes", "shocks_daily", "model_metadata")
metadata = ctx["metadata"]
cfg = ctx["cfg"]
day_map = sh.calendar_map()

episodes_all = sh.load("shock_episodes")
# Column pruning matters at full scale: the daily table holds millions of scored
# item-days, and the page only needs these columns. The per-component breakdown is
# already embedded in each episode's why_flagged payload.
daily_all = sh.load("shocks_daily", columns=[
    "item_id", "store_id", "d", "date", "score", "band",
    "dept_id", "cat_id", "state_id", "y_true", "y_pred"])

if daily_all.empty:
    st.info(
        "No series was eligible for shock scoring in this dataset - every series had "
        "too few non-zero sale days or too little out-of-sample residual history to "
        "score responsibly. Nothing is scored on noise.",
        icon=":material/info:")
    st.stop()

window_lo = day_map.get(int(daily_all["d"].min()))
window_hi = day_map.get(int(daily_all["d"].max()))
st.info(
    f"Shock detection runs on the historical window where the model forecast "
    f"out of sample: **{window_lo.date()} to {window_hi.date()}**. Scores measure "
    f"deviation from model expectation on days that already happened - they are not "
    f"a forecast of future risk. Residuals come from feature set "
    f"**{metadata['shock_config']}**, which deliberately contains no FEMA or "
    f"FRED features: if crisis signals were in the model they would be absorbed into "
    f"the prediction and disappear from the residual.",
    icon=":material/history:")

# ------------------------------------------------------------------ filters
episodes = sh.apply_filters(episodes_all, ctx)
c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    bands = st.multiselect("Severity", sh.BAND_ORDER,
                           default=["Elevated", "Severe", "Critical"])
with c2:
    classes = st.multiselect("Classification", S.CLASSIFICATIONS,
                             default=S.CLASSIFICATIONS)
with c3:
    fema_only = st.toggle("FEMA overlap only", value=False,
                          help="Show only episodes during which a FEMA declaration "
                               "was active in that state.")

if not episodes.empty:
    episodes = episodes[episodes["band"].isin(bands)
                        & episodes["classification"].isin(classes)]
    if fema_only:
        episodes = episodes[episodes["fema_overlap"]]

if episodes.empty:
    st.success("No demand-shock episodes match these filters - demand tracked the "
               "forecast within normal bounds for this selection.",
               icon=":material/check_circle:")
    st.stop()

daily = sh.apply_filters(daily_all, ctx)

# ------------------------------------------------------------------ KPIs
seed_score = float(cfg["shock"]["episode"]["seed_score"])
critical = int((episodes["band"] == "Critical").sum())
severe = int((episodes["band"] == "Severe").sum())
shock_day_share = (float((daily["score"] >= seed_score).mean())
                   if not daily.empty else float("nan"))
overlap_rate = float(episodes["fema_overlap"].mean())
top_dept = (episodes["dept_id"].value_counts().idxmax()
            if episodes["dept_id"].notna().any() else "-")

with st.container(horizontal=True):
    st.metric("Episodes", f"{len(episodes):,}", border=True)
    st.metric("Severe / critical", f"{severe:,} / {critical:,}", border=True)
    st.metric("Series-days in shock", sh.fmt_pct(shock_day_share), border=True,
              help=f"Share of all scored item-days scoring {seed_score:.0f} or above "
                   f"(Watch and worse). Reflects the sidebar filters but not the "
                   f"severity/classification controls above.")
with st.container(horizontal=True):
    st.metric("Highest score", f"{episodes['peak_score'].max():.0f}", border=True)
    st.metric("Median duration", f"{episodes['n_days'].median():.0f} days", border=True)
    st.metric("FEMA coincidence rate", sh.fmt_pct(overlap_rate), border=True,
              help="Share of episodes during which a FEMA declaration was active in "
                   "the same state. This is a coincidence rate, not a causal effect.")

if overlap_rate == 0:
    st.caption("No episode in this selection coincided with an active FEMA "
               "declaration. That is a real finding for this window, not a gap in "
               "the data - check the FEMA context panel below for what was active.")

# ------------------------------------------------------------------ charts
left, right = st.columns([3, 2])
with left:
    with st.container(border=True):
        st.markdown("**Episode timeline**")
        fig = go.Figure()
        fema_context = (sh.load("fema_context")
                        if sh.artifact_exists("fema_context") else pd.DataFrame())
        states = set(episodes["state_id"].dropna().astype(str))
        if not fema_context.empty:
            active = fema_context[
                fema_context["state_id"].astype(str).isin(states)
                & (fema_context["incident_begin"] <= window_hi)
                & (fema_context["incident_end"] >= window_lo)]
            for row in active.itertuples(index=False):
                fig.add_vrect(
                    x0=max(row.incident_begin, window_lo),
                    x1=min(row.incident_end, window_hi),
                    fillcolor=sh.AMBER, opacity=0.12, line_width=0, layer="below",
                    annotation_text=f"{row.state_id}: {row.incidentType}",
                    annotation_position="top left",
                    annotation_font=dict(size=10, color=sh.AMBER))
        TIMELINE_CAP = 3000
        plotted = episodes.nlargest(TIMELINE_CAP, "peak_score")
        for band in sh.BAND_ORDER:
            block = plotted[plotted["band"] == band]
            if block.empty:
                continue
            fig.add_trace(go.Scatter(
                x=block["start_date"], y=block["store_id"].astype(str),
                mode="markers", name=band,
                marker=dict(size=np.clip(block["n_days"] * 1.5 + 6, 6, 30),
                            color=sh.BAND_COLORS[band], opacity=0.8,
                            line=dict(width=0)),
                customdata=np.stack([block["item_id"], block["classification"],
                                     block["n_days"], block["peak_score"]], axis=-1),
                hovertemplate=("%{customdata[0]}<br>%{customdata[1]}<br>"
                               "%{customdata[2]} days, peak %{customdata[3]:.0f}"
                               "<extra></extra>")))
        fig.update_yaxes(title="Store")
        fig.update_xaxes(title="Episode start")
        st.plotly_chart(sh.style_fig(fig, 330), width="stretch")
        caption = ("Marker size is episode duration. Shaded bands mark periods when "
                   "a FEMA declaration was active in a selected state - shown for "
                   "temporal context only.")
        if len(episodes) > TIMELINE_CAP:
            caption += (f" Plotting the {TIMELINE_CAP:,} highest-scoring of "
                        f"{len(episodes):,} matching episodes to keep the chart "
                        f"readable; all of them are counted in the metrics above.")
        st.caption(caption)

with right:
    with st.container(border=True):
        st.markdown("**Classification mix**")
        counts = episodes["classification"].value_counts()
        fig = go.Figure(go.Bar(
            y=counts.index.tolist(), x=counts.to_numpy(), orientation="h",
            marker_color=[sh.CLASSIFICATION_COLORS.get(c, sh.CYAN)
                          for c in counts.index]))
        fig.update_xaxes(title="Episodes")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(sh.style_fig(fig, 330), width="stretch")

with st.container(border=True):
    st.markdown("**Shock intensity by department over time**")
    if daily.empty or daily["dept_id"].isna().all():
        st.caption("Daily scores unavailable for this selection.")
    else:
        heat = (daily.groupby(["dept_id", "date"], observed=True)["score"]
                .max().reset_index())
        pivot = heat.pivot(index="dept_id", columns="date", values="score")
        fig = go.Figure(go.Heatmap(
            z=pivot.to_numpy(), x=pivot.columns, y=pivot.index.astype(str),
            colorscale=[[0, sh.PANEL], [0.3, "#3B4754"], [0.5, "#EAB308"],
                        [0.7, "#F59E0B"], [0.85, "#F97316"], [1, "#EF4444"]],
            zmin=0, zmax=100, colorbar=dict(title="Score")))
        st.plotly_chart(sh.style_fig(fig, 240), width="stretch")
        st.caption("Highest daily shock score in each department. Dark is normal.")

# ------------------------------------------------------------------ detail
st.subheader("Why was this flagged?")
EPISODE_CAP = 300
episodes = episodes.sort_values("peak_score", ascending=False)
shortlist = episodes.head(EPISODE_CAP).reset_index(drop=True)
_lookup = sh.label_lookup()
labels = pd.Series([
    f"{_lookup.get((str(r.item_id), str(r.store_id)), str(r.item_id))} @ {r.store_id}"
    f"  -  {r.start_date:%Y-%m-%d}  ({r.classification}, peak {r.peak_score:.0f})"
    for r in shortlist.itertuples(index=False)
])
choice = st.selectbox("Episode", labels.tolist(),
                      help="Ordered by peak score, most severe first.")
if len(episodes) > EPISODE_CAP:
    st.caption(f"Showing the {EPISODE_CAP} highest-scoring of {len(episodes):,} "
               f"matching episodes. Narrow the severity, classification or sidebar "
               f"filters to inspect others.")
episode = shortlist.iloc[labels.tolist().index(choice)]
why = json.loads(episode["why_flagged"])

detail_left, detail_right = st.columns([3, 2])
with detail_left:
    with st.container(border=True):
        st.markdown(f"**{sh.describe_series(episode['item_id'], episode['store_id'], with_id=False)}** "
                    f"- {episode['classification']}")
        st.caption(f"{episode['item_id']} @ {episode['store_id']} · {sh.ANONYMITY_NOTE}")
        block = daily_all[(daily_all["item_id"] == episode["item_id"])
                          & (daily_all["store_id"] == episode["store_id"])].sort_values("d")
        fig = go.Figure()
        fig.add_vrect(x0=episode["start_date"], x1=episode["end_date"],
                      fillcolor=sh.BAND_COLORS[episode["band"]], opacity=0.16,
                      line_width=0, layer="below")
        fig.add_trace(go.Scatter(x=block["date"], y=block["y_true"], name="Actual",
                                 mode="lines", line=dict(color=sh.CYAN, width=2.2)))
        fig.add_trace(go.Scatter(x=block["date"], y=block["y_pred"], name="Forecast",
                                 mode="lines",
                                 line=dict(color=sh.AMBER, width=1.8, dash="dot")))
        fig.update_yaxes(title="Units")
        st.plotly_chart(sh.style_fig(fig, 260), width="stretch")

    with st.container(border=True):
        st.markdown("**Score anatomy**")
        comps = pd.DataFrame(why["components"])
        fig = go.Figure(go.Bar(
            y=comps["name"].str.replace("_", " ").str.capitalize(),
            x=comps["points"], orientation="h",
            marker_color=[sh.CYAN, sh.TEAL, sh.INDIGO, sh.AMBER, "#F472B6"],
            text=[f"{p:.1f}" for p in comps["points"]], textposition="outside"))
        fig.update_xaxes(title="Points contributed (of 100)")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(sh.style_fig(fig, 250), width="stretch")
        arithmetic = why["score_arithmetic"]
        steps = (f"Components sum to {arithmetic['raw_total']:.1f}, multiplied by a "
                 f"low-volume guard of {arithmetic['guard_multiplier']:.2f}")
        if arithmetic.get("warmup_cap_applied"):
            steps += (f" giving {arithmetic['before_cap']:.1f}, then capped at "
                      f"{arithmetic['warmup_cap']:.0f} because the peak day fell in "
                      f"the warm-up period")
        steps += f", giving a final score of {arithmetic['final_score']:.1f}. "
        st.caption(
            steps
            + ("The guard damped this score because the series sells at low volume; "
               "a small absolute change on a slow mover must not read as a crisis. "
               if arithmetic["guard_multiplier"] < 1 else "")
            + ("The residual dispersion was frozen at its pre-episode value so the "
               "episode could not inflate its own baseline."
               if why.get("sigma_frozen_during_episode") else ""))

with detail_right:
    with st.container(border=True):
        st.markdown("**Decision trace**")
        for step in why["rule_trace"]:
            st.markdown(f"- `{step}`")
        summary = why["residual_summary"]
        st.caption(
            f"{summary['days']} days, mean deviation {summary['mean_residual']:+.2f} "
            f"units/day, {summary['sign_share_positive']:.0%} of days above forecast.")
        if why.get("warmup"):
            st.caption("Scored during the warm-up period, so the score was capped.")

    with st.container(border=True):
        st.markdown("**Real-world context**")
        fema = json.loads(episode["fema_context"])
        if fema.get("n_events", 0) > 0:
            st.markdown(f":orange-badge[FEMA declaration active]")
            st.write(S.format_fema_sentence(fema))
            events = pd.DataFrame(fema["events"])
            columns = ["incidentType", "declarationType", "incident_begin",
                       "incident_end", "counties_designated", "overlap_days"]
            for flag, label in (("end_clipped", "Ongoing at window end"),
                                ("end_estimated", "End date estimated")):
                if flag in events.columns and events[flag].any():
                    events[label] = events[flag]
                    columns.append(label)
            st.dataframe(events[columns], hide_index=True, width="stretch")
        else:
            st.write(S.format_fema_sentence(fema))

        econ = json.loads(episode["econ_context"])
        if econ.get("available"):
            change = econ.get("change_vs_3m_ago")
            change_txt = (f", {change:+.1f} points versus three months earlier"
                          if change is not None else "")
            st.caption(
                f"State unemployment in {econ['state']} was "
                f"{econ['unemployment_rate']:.1f}% ({econ['source_month']})"
                f"{change_txt}. {econ['disclaimer']}")

with st.expander("How the demand shock score is built", icon=":material/functions:"):
    weights = cfg["shock"]["weights"]
    st.markdown(
        "Each day, five signals are computed from the forecast residual "
        "`e = actual - forecast`, squashed to a 0-1 range by a piecewise-linear cap, "
        "then weighted to a 0-100 score:")
    st.dataframe(pd.DataFrame([
        {"Signal": "Standardised residual",
         "Weight": weights["std_residual"],
         "Measures": "Deviation relative to this series' own recent forecast error"},
        {"Signal": "Percentage deviation", "Weight": weights["pct_deviation"],
         "Measures": "Deviation relative to recent average demand"},
        {"Signal": "Persistence", "Weight": weights["persistence"],
         "Measures": "Consecutive days deviating in the same direction"},
        {"Signal": "Volatility ratio", "Weight": weights["volatility_ratio"],
         "Measures": "Short-term error spread against the longer-term spread"},
        {"Signal": "Level shift", "Weight": weights["level_shift"],
         "Measures": "Change in the demand level itself, not just the error"},
    ]), hide_index=True, width="stretch")
    st.markdown(
        f"The raw total is multiplied by a low-volume guard "
        f"(minimum {cfg['shock']['guard_min_multiplier']}) so a small absolute change "
        f"on a slow-moving item cannot produce a high score, and a zero-sale day "
        f"against a near-zero forecast is forced to zero. Series with fewer than "
        f"{cfg['shock']['eligibility']['min_nonzero_days_trailing_56']} non-zero days "
        f"in the trailing 56 days are excluded rather than scored on noise.\n\n"
        f"**Severity bands:** " + " - ".join(
            f"{name} {int(lo)}-{int(hi) if hi <= 100 else 100}"
            for name, (lo, hi) in cfg["shock"]["bands"].items()))
    st.warning("The demand shock score is a metric defined by this project. It is "
               "not a validated industry standard.", icon=":material/info:")
