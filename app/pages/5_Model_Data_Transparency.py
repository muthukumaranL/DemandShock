"""Module 5 - Model performance, explainability and data quality."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shared as sh
import streamlit as st

from demandshock import features as F  # noqa: E402

sh.page_setup("Model, explainability & data quality", ":material/verified:")
# Every figure on this page is model-wide by definition, so no hierarchy filters.
ctx = sh.sidebar_context(filters=False)
sh.header("MODEL, EXPLAINABILITY &amp; DATA QUALITY",
          "Can this model be trusted, why does it predict what it does, and is the "
          "underlying data sound?")

sh.require_page("model_metadata")
metadata = ctx["metadata"]
selected = metadata["selected_config"]

tab_perf, tab_explain, tab_quality = st.tabs(
    ["Performance & ablation", "Explainability", "Data quality"])

# ===========================================================================
with tab_perf:
    st.markdown(
        f"**Strategy.** {metadata.get('strategy', '')}. Objective "
        f"`{metadata.get('objective')}`, {metadata.get('n_features')} features, "
        f"{metadata.get('training_rows_deploy', 0):,} training rows in the deploy fit.")
    st.caption(
        "All horizons share one information set frozen 28 days before the target "
        "date. A 7-day-ahead forecast therefore does not see the most recent week - "
        "the trade for having a single model with no recursive feedback loop, and "
        "residuals that mean exactly one thing.")

    if not sh.artifact_exists("ablation"):
        sh.missing_section("ablation", "Ablation results")
    else:
        ablation = sh.load("ablation")
        horizon = int(ctx["cfg"]["ablation"]["selection_horizon"])
        at_h = ablation[ablation["horizon"] == horizon]
        summary = (at_h.groupby("config")
                   .agg(wape=("wape", "mean"), wape_std=("wape", "std"),
                        rmsse=("rmsse", "mean"), bias=("bias", "mean"),
                        n_features=("n_features", "first"),
                        train_seconds=("train_seconds", "mean"))
                   .reindex(["A", "B", "C", "D"]).dropna(how="all").reset_index())
        summary["label"] = summary["config"].map(F.ABLATION_LABELS)
        best = summary.loc[summary["wape"].idxmin(), "config"]

        st.subheader("Does external data actually improve the forecast?")
        st.dataframe(
            summary[["config", "label", "n_features", "wape", "wape_std", "rmsse",
                     "bias"]],
            hide_index=True, width="stretch",
            column_config={
                "config": st.column_config.TextColumn("Arm"),
                "label": st.column_config.TextColumn("Feature set"),
                "n_features": st.column_config.NumberColumn("Features"),
                "wape": st.column_config.NumberColumn("WAPE (mean)", format="%.4f"),
                "wape_std": st.column_config.NumberColumn("WAPE (sd across folds)",
                                                          format="%.4f"),
                "rmsse": st.column_config.NumberColumn("RMSSE", format="%.4f"),
                "bias": st.column_config.NumberColumn("Bias", format="%+.4f"),
            })
        st.caption(f"Mean across the rolling-origin folds at horizon {horizon}. "
                   f"Identical seed, parameters, folds and rows in every arm.")

        # honest, auto-generated verdicts
        by_config = summary.set_index("config")
        pivot = at_h.pivot_table(index="fold_id", columns="config", values="wape")

        def verdict(arm: str, previous: str, name: str) -> None:
            if arm not in by_config.index or previous not in by_config.index:
                return
            delta = float(by_config.loc[arm, "wape"] - by_config.loc[previous, "wape"])
            rel = delta / float(by_config.loc[previous, "wape"])
            spread = float(by_config.loc[previous, "wape_std"] or 0)
            wins = int((pivot[arm] < pivot[previous]).sum()) if arm in pivot and previous in pivot else 0
            folds = int(pivot[arm].notna().sum()) if arm in pivot else 0
            if delta < 0:
                message = (f"**{name} improved accuracy**: WAPE fell by "
                           f"{abs(delta):.4f} ({abs(rel):.2%} relative), better on "
                           f"{wins} of {folds} folds.")
                if abs(delta) < spread:
                    message += (f" This is smaller than the fold-to-fold spread of "
                                f"{spread:.4f}, so treat it as a marginal gain rather "
                                f"than a decisive one.")
                st.success(message, icon=":material/trending_down:")
            else:
                st.warning(
                    f"**{name} did not improve accuracy**: WAPE rose by {delta:.4f} "
                    f"({rel:.2%} relative), better on only {wins} of {folds} folds. "
                    f"These features are retained for explaining shocks in module 3, "
                    f"where they add context, not for point-forecast accuracy.",
                    icon=":material/trending_up:")

        verdict("B", "A", "Calendar and price features")
        verdict("C", "B", "FEMA disaster features")
        verdict("D", "C", "FRED economic features")

        st.caption(
            f"Selection rule: the smallest feature set within "
            f"{ctx['cfg']['ablation']['selection_tolerance_rel']:.1%} relative of the "
            f"best mean WAPE. Selected **{selected}**"
            + (f" (best raw score was {best})." if best != selected else "."))

        importance_note = None
        if sh.artifact_exists("feature_importance"):
            fi = sh.load("feature_importance")
            # Attribution comes from the SELECTED model only. If FEMA features are
            # not in that feature set, a "0.00% of gain" figure would be a structural
            # certainty masquerading as a measurement - say what is actually true.
            if "fema" in set(fi["family"]):
                fema_share = float(fi[fi["family"] == "fema"]["gain"].sum()
                                   / max(fi["gain"].sum(), 1e-9))
                importance_note = (
                    f"For perspective: FEMA features account for {fema_share:.2%} of "
                    f"total model gain. A small WAPE difference alongside a share this "
                    f"low is weak evidence that disaster context improves point "
                    f"forecasts.")
            else:
                importance_note = (
                    "FEMA features are not part of the selected feature set, so they "
                    "carry no attribution here by construction. The ablation table "
                    "above is the evidence on whether they help.")
        if importance_note:
            st.caption(importance_note)

        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.markdown("**Accuracy by feature set and horizon**")
                fig = go.Figure()
                for config in sorted(ablation["config"].unique()):
                    block = (ablation[ablation["config"] == config]
                             .groupby("horizon", as_index=False)["wape"].mean())
                    fig.add_trace(go.Bar(x=block["horizon"].astype(str), y=block["wape"],
                                         name=f"{config}"))
                fig.update_yaxes(title="WAPE", tickformat=".1%")
                # Categorical, else Plotly interpolates a "21" tick that is not a
                # horizon this project evaluates.
                fig.update_xaxes(title="Horizon (days)", type="category")
                st.plotly_chart(sh.style_fig(fig, 300), width="stretch")
        with right:
            with st.container(border=True):
                st.markdown("**Per-fold stability**")
                fig = go.Figure()
                for config in sorted(at_h["config"].unique()):
                    block = at_h[at_h["config"] == config].sort_values("fold_id")
                    fig.add_trace(go.Scatter(x=block["fold_id"], y=block["wape"],
                                             mode="lines+markers", name=config))
                fig.update_yaxes(title="WAPE", tickformat=".1%")
                st.plotly_chart(sh.style_fig(fig, 300), width="stretch")
                st.caption("If the gap between arms is smaller than the movement "
                           "between folds, the difference is not decisive.")

    if sh.artifact_exists("metrics"):
        metrics = sh.load("metrics")
        st.subheader("Held-out performance against the benchmarks")
        block = metrics[(metrics["fold_id"] == "HOLDOUT")
                        & (metrics["level"] == "overall") & (metrics["horizon"] == 28)]
        block = block[(block["model"] != "lgbm") | (block["config"] == selected)]
        table = block[["model", "mae", "rmse", "rmsse", "wape", "smape", "bias",
                       "n_series", "n_rmsse_excluded"]].copy()
        table["model"] = table["model"].map(lambda m: sh.MODEL_LABELS.get(m, m))
        st.dataframe(table.sort_values("wape"), hide_index=True, width="stretch",
                     column_config={
                         "model": st.column_config.TextColumn("Model"),
                         "wape": st.column_config.NumberColumn("WAPE", format="%.4f"),
                         "rmsse": st.column_config.NumberColumn("RMSSE", format="%.4f"),
                         "bias": st.column_config.NumberColumn("Bias", format="%+.4f"),
                         "smape": st.column_config.NumberColumn("sMAPE", format="%.1f"),
                         "n_rmsse_excluded": st.column_config.NumberColumn(
                             "RMSSE excluded"),
                     })
        st.caption(
            "sMAPE is shown for completeness but is misleading here: on a zero-sale "
            "day a naive forecast of exactly zero scores perfectly while any positive "
            "forecast is penalised the full 200%. With over half of item-days at zero, "
            "the naive models flatter themselves on this metric. WAPE and RMSSE are "
            "the trustworthy comparisons.")

        coverage = metadata.get("interval_coverage_holdout") or {}
        if coverage:
            st.metric("P10-P90 interval coverage on the holdout",
                      f"{coverage.get('coverage', float('nan')):.1%}", border=True,
                      help="Target is 80% by construction. Measured on the untouched "
                           "holdout window.")
            st.caption(
                f"{coverage.get('below', 0):.1%} of actuals fell below P10 and "
                f"{coverage.get('above', 0):.1%} above P90, across "
                f"{coverage.get('n', 0):,} item-days. Intervals come from empirical "
                f"out-of-sample residuals grouped by store, category and forecast level.")

    st.caption(
        "Disclosure: fold F1 was used for the objective spot-check (Tweedie vs "
        "Poisson vs squared error, development mode only) before the objective was "
        "frozen, and it also contributes to the cross-fold means above. No other "
        "hyperparameter search was run. The holdout window was opened once, after "
        "feature-set selection, and never used for tuning.")

# ===========================================================================
with tab_explain:
    if not sh.artifact_exists("feature_importance"):
        sh.missing_section("feature_importance", "Feature attribution")
    else:
        fi = sh.load("feature_importance")
        shap_available = metadata.get("shap_available", False)
        st.caption(
            "Attribution values are exact TreeSHAP contributions computed natively by "
            "LightGBM through `predict(pred_contrib=True)`"
            + ("." if shap_available else
               ", so the optional `shap` package is not required and its absence "
               "changes nothing about what is shown here."))

        left, right = st.columns([3, 2])
        with left:
            with st.container(border=True):
                st.markdown("**What drives the forecast**")
                top_n = int(ctx["cfg"]["explainability"]["top_features"])
                top = fi.nlargest(top_n, "gain").sort_values("gain")
                family_colors = {"demand": sh.CYAN, "calendar": sh.TEAL,
                                 "price": sh.INDIGO, "fema": sh.AMBER,
                                 "fred": sh.RED, "static": sh.MUTED}
                fig = go.Figure(go.Bar(
                    y=top["feature"], x=top["gain"], orientation="h",
                    marker_color=[family_colors.get(f, sh.CYAN) for f in top["family"]]))
                fig.update_xaxes(title="Total gain")
                st.plotly_chart(sh.style_fig(fig, 460), width="stretch")
        with right:
            with st.container(border=True):
                st.markdown("**Share of model gain by family**")
                share = (fi.groupby("family")["gain"].sum()
                         / max(fi["gain"].sum(), 1e-9)).sort_values(ascending=False)
                labels = [F.FAMILY_LABELS.get(f, f) for f in share.index]
                fig = go.Figure(go.Bar(
                    y=labels, x=share.to_numpy(), orientation="h",
                    marker_color=[family_colors.get(f, sh.CYAN) for f in share.index],
                    text=[f"{v:.1%}" for v in share], textposition="outside"))
                fig.update_xaxes(title="Share of total gain", tickformat=".0%")
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(sh.style_fig(fig, 300), width="stretch")

            with st.container(border=True):
                st.markdown("**In plain language**")
                leader = fi.iloc[0]
                demand_share = float(share.get("demand", 0))
                families_present = set(fi["family"])
                external = [f for f in ("fema", "fred") if f in families_present]
                if external:
                    external_line = (
                        f"- External context ({' and '.join(f.upper() for f in external)})"
                        f" contributes "
                        f"{sum(share.get(f, 0) for f in external):.2%}, which is why "
                        f"the ablation verdict above should be read carefully.")
                else:
                    external_line = (
                        "- External context (FEMA and FRED) is not in the selected "
                        "feature set, so it has no attribution here by construction - "
                        "read the ablation verdict above for whether it helps.")
                st.markdown(
                    f"- Recent demand history dominates: it accounts for "
                    f"{demand_share:.0%} of total model gain, led by "
                    f"`{leader['feature']}`.\n"
                    f"- Calendar effects contribute {share.get('calendar', 0):.1%} - "
                    f"weekday and SNAP timing shift grocery demand systematically.\n"
                    f"- Price features contribute {share.get('price', 0):.1%}, mainly "
                    f"price relative to a product's own recent average.\n"
                    + external_line)
                st.caption("Attribution measures how much the model relies on each "
                           "feature, not the direction of its effect - these values "
                           "are unsigned. It does not establish that any feature "
                           "causes demand to change.")

        with st.container(border=True):
            st.markdown("**Full attribution table**")
            table = fi[["rank", "feature", "family", "gain", "split",
                        "mean_abs_contrib"]].copy()
            table["family"] = table["family"].map(
                lambda f: F.FAMILY_LABELS.get(f, f))
            st.dataframe(table, hide_index=True, width="stretch", height=300,
                         column_config={
                             "gain": st.column_config.NumberColumn("Gain", format="%.0f"),
                             "split": st.column_config.NumberColumn("Splits"),
                             "mean_abs_contrib": st.column_config.NumberColumn(
                                 "Mean |contribution|", format="%.4f",
                                 help="Average absolute exact-TreeSHAP contribution "
                                      "per prediction, in model output units."),
                         })

# ===========================================================================
with tab_quality:
    if not sh.artifact_exists("data_quality"):
        sh.missing_section("data_quality", "Data-quality report")
    else:
        quality = sh.load_json("data_quality")
        tables = quality.get("tables", {})
        checks = pd.DataFrame(quality.get("checks", []))

        n_pass = int((checks["status"] == "pass").sum()) if not checks.empty else 0
        n_warn = int((checks["status"] == "warn").sum()) if not checks.empty else 0
        n_fail = int((checks["status"] == "fail").sum()) if not checks.empty else 0
        with st.container(horizontal=True):
            st.metric("Checks passed", f"{n_pass}", border=True)
            st.metric("Warnings", f"{n_warn}", border=True)
            st.metric("Failures", f"{n_fail}", border=True)
        if n_fail == 0:
            st.success("Every data-quality check passed on the last pipeline run.",
                       icon=":material/check_circle:")

        st.subheader("Coverage")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Series", sh.fmt_units(tables.get("series")))
        c1.metric("Items", sh.fmt_units(tables.get("items")))
        c2.metric("Stores", sh.fmt_units(tables.get("stores")))
        c2.metric("Departments", sh.fmt_units(tables.get("departments")))
        c3.metric("Item-days", sh.fmt_units(tables.get("sales_rows")))
        c3.metric("Price records", sh.fmt_units(tables.get("price_rows")))
        c4.metric("FEMA disasters", sh.fmt_units(tables.get("fema_disasters")))
        c4.metric("FRED observations", sh.fmt_units(tables.get("fred_months")))
        st.caption(
            f"{tables.get('date_start', '?')} to {tables.get('date_end', '?')} - "
            f"{tables.get('calendar_days', '?')} calendar days - "
            f"{tables.get('zero_demand_ratio', 0):.1%} of item-days have zero sales.")

        st.subheader("Every check")
        if not checks.empty:
            display = checks.copy()
            display["expected"] = display["expected"].astype(str).str.slice(0, 120)
            display["observed"] = display["observed"].astype(str).str.slice(0, 120)
            status_filter = st.multiselect(
                "Show", ["pass", "warn", "fail"], default=["pass", "warn", "fail"])
            display = display[display["status"].isin(status_filter)]
            st.dataframe(
                display[["status", "name", "expected", "observed", "detail"]],
                hide_index=True, width="stretch", height=420,
                column_config={
                    "status": st.column_config.TextColumn("Status", width="small"),
                    "name": st.column_config.TextColumn("Check"),
                    "detail": st.column_config.TextColumn("Notes", width="large"),
                })

        st.subheader("Known limitations")
        st.markdown(
            "- **No inventory data exists in M5.** On-hand stock, lead times, service "
            "levels and costs are user inputs, never derived or invented.\n"
            "- **FEMA is county-scoped; M5 gives only a store's state.** Disaster "
            "context is therefore a state-level temporal coincidence and is never "
            "presented as a cause.\n"
            "- **FRED unemployment is monthly** and is joined with a publication lag "
            "so the model cannot see a figure before it was released. It cannot "
            "explain a daily deviation.\n"
            "- **Shock detection is retrospective.** Residuals exist only where the "
            "model forecast out of sample, so shocks are measured on days that "
            "already happened.\n"
            "- **The official M5 WRMSSE is not implemented**, so that metric is never "
            "quoted. Per-series RMSSE and its aggregates are used instead.\n"
            "- **Weather data is out of scope** in this version and no component "
            "depends on it.\n"
            "- The demand shock score is a **project-defined metric**, not an "
            "industry standard.")

        with st.expander("Pipeline provenance", icon=":material/inventory:"):
            st.json({
                "mode": metadata.get("mode"),
                "config_hash": metadata.get("config_hash"),
                "trained_at": metadata.get("trained_at"),
                "selected_feature_set": metadata.get("selected_config"),
                "shock_residual_source": metadata.get("shock_config"),
                "train_window_days": metadata.get("train_window_days"),
                "folds": metadata.get("folds"),
                "library_versions": metadata.get("library_versions"),
                "shap_package_available": metadata.get("shap_available"),
            })
