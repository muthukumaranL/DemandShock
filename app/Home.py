"""DemandShock - landing page."""

from __future__ import annotations

import shared as sh
import streamlit as st

sh.page_setup("Home", ":material/radar:")
ctx = sh.sidebar_context()

sh.header("DEMANDSHOCK",
          "Crisis-aware demand forecasting &amp; inventory intelligence")

st.markdown(
    "Traditional forecasting asks *what will demand be?* DemandShock also asks "
    "**where is actual demand departing from expectation, what real-world "
    "conditions coincided with it, and what should a planner do about it?**")

metadata = ctx.get("metadata") or {}
if not metadata:
    st.warning("No trained model found yet. Build the artifacts first:",
               icon=":material/build:")
    st.code(sh.REBUILD_COMMAND, language="bash")
    st.stop()

quality = sh.load_json("data_quality") if sh.artifact_exists("data_quality") else {}
tables = quality.get("tables", {})

with st.container(horizontal=True):
    st.metric("Real series", sh.fmt_units(tables.get("series")), border=True)
    st.metric("Item-days processed", sh.fmt_units(tables.get("sales_rows")), border=True)
    st.metric("FEMA disasters", sh.fmt_units(tables.get("fema_disasters")), border=True)
    st.metric("Feature set in use", metadata.get("selected_config", "-"), border=True)

st.caption(
    f"Coverage {tables.get('date_start', '?')} to {tables.get('date_end', '?')} - "
    f"{tables.get('stores', '?')} stores, {tables.get('categories', '?')} categories, "
    f"{tables.get('items', '?')} items - "
    f"{tables.get('zero_demand_ratio', 0):.0%} of item-days have zero sales.")

st.subheader("The five modules")
col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("**:material/dashboard: 1 - Executive command centre**")
        st.caption("What needs management attention right now: accuracy, bias, "
                   "severe demand shocks and dollar exposure.")
    with st.container(border=True):
        st.markdown("**:material/query_stats: 2 - Forecasting intelligence**")
        st.caption("Forecast versus actual by state, store, category and item, with "
                   "uncertainty bands and model comparison.")
    with st.container(border=True):
        st.markdown("**:material/crisis_alert: 3 - Crisis &amp; demand shock intelligence**")
        st.caption("Where demand broke from expectation, how severe, and which real "
                   "FEMA declarations coincided with it.")
with col2:
    with st.container(border=True):
        st.markdown("**:material/inventory_2: 4 - Inventory &amp; business impact**")
        st.caption("Forecast-derived planning requirements and dollar exposure. "
                   "Planning parameters require your own operating inputs.")
    with st.container(border=True):
        st.markdown("**:material/verified: 5 - Model, explainability &amp; data quality**")
        st.caption("Honest performance, the FEMA/FRED ablation, feature attribution "
                   "and every data-quality check.")

st.subheader("How to read this platform")
sh.note(
    "<b>Real data only.</b> Every number comes from the M5 competition dataset, "
    "FEMA disaster declarations and FRED state unemployment. Nothing is simulated, "
    "and no metric shown anywhere is a placeholder.<br><br>"
    "<b>Context, not causation.</b> When a demand shock coincides with a FEMA "
    "declaration, the platform reports the coincidence and says so explicitly. It "
    "never claims the event caused the change in demand.<br><br>"
    "<b>Retrospective by design.</b> M5 ends on 2016-06-19. Shock detection runs on "
    "the historical window where the model forecast out of sample, so every shock "
    "shown is a real, measured deviation - not a projection.")

st.subheader("Known limitations")
st.markdown(
    "- M5 contains **no inventory records**. Safety stock, reorder points and days "
    "of cover therefore require operating assumptions you enter yourself.\n"
    "- FEMA declarations are **county-scoped** while M5 discloses only a store's "
    "state, so disaster context is state-level association only.\n"
    "- FRED unemployment is **monthly**; it cannot explain a daily deviation and is "
    "used as slow-moving background context.\n"
    "- Weather data is deliberately **out of scope** in this version.\n"
    "- The demand shock score is a **project-defined metric**, not an industry standard.")
