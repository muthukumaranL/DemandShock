"""Shared UI layer: theme, cached artifact access, filters, graceful degradation.

The application NEVER trains, never reads raw CSVs and never recomputes anything
expensive. It reads the artifacts produced by scripts/run_pipeline.py, which is
what keeps the UI responsive regardless of how large the training run was.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from demandshock.config import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------
INK = "#0F1419"
PANEL = "#161D26"
BORDER = "#232B36"
TEXT = "#E6EDF3"
MUTED = "#8B98A5"
CYAN = "#22D3EE"
TEAL = "#14B8A6"
INDIGO = "#818CF8"
AMBER = "#F59E0B"
RED = "#EF4444"
GREEN = "#10B981"

BAND_COLORS = {
    "Normal": "#3B4754",
    "Watch": "#EAB308",
    "Elevated": "#F59E0B",
    "Severe": "#F97316",
    "Critical": "#EF4444",
}
BAND_ORDER = ["Normal", "Watch", "Elevated", "Severe", "Critical"]

CLASSIFICATION_COLORS = {
    "Demand Surge": CYAN,
    "Demand Collapse": RED,
    "Volatility Shock": INDIGO,
    "Persistent Under-forecast": TEAL,
    "Persistent Over-forecast": AMBER,
    "Regime Shift": "#F472B6",
    "Possible Regime Shift (window truncated)": "#C084FC",
}

MODEL_LABELS = {
    "naive": "Naive (last day)",
    "snaive7": "Seasonal naive (lag 7)",
    "snaive28": "Seasonal naive (lag 28)",
    "lgbm": "LightGBM",
}

_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=13),
        colorway=[CYAN, TEAL, INDIGO, AMBER, RED, MUTED],
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=0),
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER,
                        font=dict(color=TEXT, size=12)),
    )
)
pio.templates["demandshock"] = _TEMPLATE

_CSS = f"""
<style>
  [data-testid="stMetric"] {{
      background: {PANEL};
      border: 1px solid {BORDER};
      border-left: 3px solid {CYAN};
      border-radius: 8px;
      padding: 14px 16px;
  }}
  [data-testid="stMetricLabel"] p {{
      color: {MUTED};
      font-size: 0.78rem;
      letter-spacing: .04em;
      text-transform: uppercase;
  }}
  .ds-header {{
      border-bottom: 1px solid {BORDER};
      padding-bottom: 10px;
      margin-bottom: 4px;
  }}
  .ds-header h1 {{
      font-size: 1.55rem; font-weight: 650; letter-spacing: .10em;
      margin: 0; color: {TEXT};
  }}
  .ds-header .ds-sub {{ color: {MUTED}; font-size: .92rem; margin-top: 2px; }}
  .ds-userinput {{
      display:inline-block; background:#3A2E10; color:{AMBER};
      border:1px solid {AMBER}; border-radius:5px;
      padding:1px 8px; font-size:.7rem; letter-spacing:.06em; font-weight:600;
  }}
  .ds-note {{
      background: {PANEL}; border-left: 3px solid {MUTED};
      border-radius: 6px; padding: 10px 14px; color: {MUTED}; font-size: .86rem;
  }}
</style>
"""


def page_setup(title: str, icon: str) -> None:
    st.set_page_config(page_title=f"DemandShock - {title}", page_icon=icon,
                       layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)


def header(title: str, question: str) -> None:
    st.markdown(
        f'<div class="ds-header"><h1>{title}</h1>'
        f'<div class="ds-sub">{question}</div></div>',
        unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="ds-note">{text}</div>', unsafe_allow_html=True)


def user_input_badge() -> str:
    return '<span class="ds-userinput">USER INPUT</span>'


def style_fig(fig: go.Figure, height: int = 320, title: str | None = None) -> go.Figure:
    fig.update_layout(template="demandshock", height=height)
    # Passing title=None makes Plotly render the literal string "undefined" as the
    # chart title, so set the text explicitly in both branches.
    fig.update_layout(title=dict(text=title or "", font=dict(size=14)))
    return fig


# ---------------------------------------------------------------------------
# config + artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def get_config():
    return load_config(REPO_ROOT / "config.yaml")


ARTIFACT_FILES = {
    "metrics": "metrics.parquet",
    "ablation": "ablation.parquet",
    "shocks_daily": "shocks_daily.parquet",
    "shock_episodes": "shock_episodes.parquet",
    "inventory_base": "inventory_base.parquet",
    "feature_importance": "feature_importance.parquet",
    "residual_quantiles": "residual_quantiles.parquet",
}
PROCESSED_FILES = {
    "calendar": "calendar.parquet",
    "series_meta": "series_meta.parquet",
    "prices": "prices_long.parquet",
    "fema_context": "fema_context.parquet",
    "fred_daily": "fred_state_daily.parquet",
}
JSON_FILES = {
    "model_metadata": "model_metadata.json",
    "data_quality": "data_quality.json",
    "fold_definitions": "fold_definitions.json",
}


@st.cache_data(show_spinner=False, max_entries=64)
def _read_parquet(path_str: str, mtime: float, columns: tuple[str, ...] | None = None):
    return pd.read_parquet(path_str, columns=list(columns) if columns else None)


@st.cache_data(show_spinner=False, max_entries=16)
def _read_json(path_str: str, mtime: float) -> dict:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def _resolve(name: str) -> Path:
    cfg = get_config()
    if name in ARTIFACT_FILES:
        return cfg.artifacts_dir / ARTIFACT_FILES[name]
    if name in PROCESSED_FILES:
        return cfg.processed_dir / PROCESSED_FILES[name]
    if name in JSON_FILES:
        return cfg.artifacts_dir / JSON_FILES[name]
    raise KeyError(f"Unknown artifact {name!r}")


def artifact_exists(name: str) -> bool:
    try:
        return _resolve(name).exists()
    except KeyError:
        return False


def load(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a tabular artifact. Cache invalidates automatically on rebuild."""
    path = _resolve(name)
    if not path.exists():
        raise FileNotFoundError(name)
    return _read_parquet(str(path), path.stat().st_mtime,
                         tuple(columns) if columns else None)


def load_json(name: str) -> dict:
    path = _resolve(name)
    if not path.exists():
        raise FileNotFoundError(name)
    return _read_json(str(path), path.stat().st_mtime)


@st.cache_data(show_spinner=False, max_entries=48)
def _read_forecasts(root: str, mtime: float, model: str | None, config: str | None,
                    fold: str | None) -> pd.DataFrame:
    import pyarrow.dataset as ds

    dataset = ds.dataset(root, format="parquet", partitioning="hive")
    filt = None
    for field, value in (("model", model), ("config", config), ("fold_id", fold)):
        if value is None:
            continue
        clause = ds.field(field) == value
        filt = clause if filt is None else (filt & clause)
    frame = dataset.to_table(filter=filt).to_pandas()
    return frame


def load_forecasts(model: str | None = None, config: str | None = None,
                   fold: str | None = None) -> pd.DataFrame:
    """Read one partition slice of the forecast store - never the whole thing."""
    cfg = get_config()
    root = cfg.artifacts_dir / "forecasts.parquet"
    if not root.exists():
        raise FileNotFoundError("forecasts")
    newest = max((p.stat().st_mtime for p in root.rglob("*.parquet")), default=0.0)
    return _read_forecasts(str(root), newest, model, config, fold)


@st.cache_data(show_spinner=False, max_entries=8)
def _read_sales(root: str, mtime: float, d_min: int | None) -> pd.DataFrame:
    import pyarrow.dataset as ds

    dataset = ds.dataset(root, format="parquet", partitioning="hive")
    filt = (ds.field("d") >= d_min) if d_min is not None else None
    return dataset.to_table(filter=filt).to_pandas()


def load_sales(d_min: int | None = None) -> pd.DataFrame:
    cfg = get_config()
    root = cfg.processed_dir / "sales_long"
    if not root.exists():
        raise FileNotFoundError("sales_long")
    newest = max((p.stat().st_mtime for p in root.rglob("*.parquet")), default=0.0)
    return _read_sales(str(root), newest, d_min)


@st.cache_data(show_spinner=False)
def day_to_date_map(_mtime: float) -> dict[int, pd.Timestamp]:
    cal = load("calendar", ["d", "date"])
    return dict(zip(cal["d"], cal["date"]))


def calendar_map() -> dict[int, pd.Timestamp]:
    return day_to_date_map(_resolve("calendar").stat().st_mtime)


# ---------------------------------------------------------------------------
# graceful degradation
# ---------------------------------------------------------------------------
REBUILD_COMMAND = "python scripts/run_pipeline.py --mode development"

PRODUCER = {
    "metrics": "scripts/train.py",
    "ablation": "scripts/train.py",
    "feature_importance": "scripts/train.py",
    "residual_quantiles": "scripts/train.py",
    "model_metadata": "scripts/train.py",
    "shocks_daily": "scripts/evaluate.py",
    "shock_episodes": "scripts/evaluate.py",
    "inventory_base": "scripts/evaluate.py",
    "calendar": "scripts/prepare_data.py",
    "series_meta": "scripts/prepare_data.py",
    "prices": "scripts/prepare_data.py",
    "fema_context": "scripts/prepare_data.py",
    "fred_daily": "scripts/prepare_data.py",
    "data_quality": "scripts/prepare_data.py",
}


def require(*names: str) -> bool:
    """Render an actionable panel instead of a traceback when artifacts are absent."""
    missing = [n for n in names if not artifact_exists(n)]
    if not missing:
        return True
    st.warning(
        f"This view needs artifacts that have not been built yet: "
        f"**{', '.join(missing)}**.",
        icon=":material/build:")
    producers = sorted({PRODUCER.get(n, "scripts/run_pipeline.py") for n in missing})
    st.caption(f"Produced by: {', '.join(producers)}")
    st.code(REBUILD_COMMAND, language="bash")
    return False


def require_page(*names: str) -> bool:
    """Same as require(), but stops the page so nothing downstream can error."""
    if require(*names):
        return True
    st.stop()
    return False


def missing_section(name: str, what: str) -> None:
    st.warning(f"{what} is unavailable - `{ARTIFACT_FILES.get(name, name)}` not found "
               f"(produced by `{PRODUCER.get(name, 'scripts/run_pipeline.py')}`).",
               icon=":material/info:")


def empty_filters(message: str = "No data matches the current filters.") -> None:
    st.info(message + " Widen the selection in the sidebar.",
            icon=":material/filter_alt_off:")


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
def sidebar_context() -> dict[str, Any]:
    """Mode badge, pipeline status and the shared hierarchy filters."""
    cfg = get_config()
    ctx: dict[str, Any] = {"cfg": cfg}

    with st.sidebar:
        st.markdown("### DemandShock")
        metadata = {}
        if artifact_exists("model_metadata"):
            metadata = load_json("model_metadata")
        ctx["metadata"] = metadata

        mode = metadata.get("mode", cfg.mode)
        label = "Development subset" if mode == "development" else "Full dataset"
        st.markdown(
            f":blue-badge[{label}] "
            + (":green-badge[Model ready]" if metadata else ":red-badge[No model]"))

        if metadata:
            st.caption(f"Trained {metadata.get('trained_at', 'unknown')} - "
                       f"feature set {metadata.get('selected_config', '?')} - "
                       f"{metadata.get('n_features', '?')} features")
            # Compare against the config as it would be for the artifacts' OWN mode:
            # a full-mode artifact set is not "stale" merely because config.yaml
            # currently defaults to development.
            if metadata.get("config_hash"):
                comparable = load_config(REPO_ROOT / "config.yaml", mode=mode).hash()
                if metadata["config_hash"] != comparable:
                    st.warning("config.yaml has changed since these artifacts were "
                               "built. Re-run the pipeline to refresh them.",
                               icon=":material/sync_problem:")
            if metadata.get("reduced_ablation_run"):
                st.warning("Artifacts come from a reduced (smoke) ablation run and "
                           "must not be quoted as ablation results.",
                           icon=":material/warning:")

        if artifact_exists("series_meta"):
            meta = load("series_meta")
            ctx["meta"] = meta
            st.markdown("#### Filters")
            states = sorted(meta["state_id"].unique())
            state = st.multiselect("State", states, default=states)
            stores = sorted(meta[meta["state_id"].isin(state)]["store_id"].unique())
            store = st.multiselect("Store", stores, default=stores)
            cats = sorted(meta[meta["store_id"].isin(store)]["cat_id"].unique())
            category = st.multiselect("Category", cats, default=cats)
            depts = sorted(
                meta[meta["store_id"].isin(store)
                     & meta["cat_id"].isin(category)]["dept_id"].unique())
            dept = st.multiselect("Department", depts, default=depts)
            ctx["filters"] = {"state_id": state, "store_id": store,
                              "cat_id": category, "dept_id": dept}
            selected = meta[
                meta["state_id"].isin(state) & meta["store_id"].isin(store)
                & meta["cat_id"].isin(category) & meta["dept_id"].isin(dept)]
            ctx["selected_series"] = selected
            st.caption(f"{len(selected):,} of {len(meta):,} series selected")

        with st.expander("Advanced", icon=":material/tune:"):
            if st.button("Clear cached data", width="stretch"):
                st.cache_data.clear()
                st.rerun()
            st.caption(f"Artifacts: `{cfg.artifacts_dir.name}/`")
    return ctx


def apply_filters(frame: pd.DataFrame, ctx: dict[str, Any]) -> pd.DataFrame:
    """Restrict a frame to the sidebar selection using whichever keys it carries."""
    selected = ctx.get("selected_series")
    if selected is None or frame.empty:
        return frame
    if {"item_id", "store_id"}.issubset(frame.columns):
        keys = set(zip(selected["item_id"].astype(str), selected["store_id"].astype(str)))
        pairs = list(zip(frame["item_id"].astype(str), frame["store_id"].astype(str)))
        return frame[[p in keys for p in pairs]]
    for col in ("store_id", "state_id", "cat_id", "dept_id"):
        if col in frame.columns:
            allowed = set(selected[col].astype(str))
            return frame[frame[col].astype(str).isin(allowed)]
    return frame


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
MAX_PICKER_OPTIONS = 400


@st.cache_data(show_spinner=False, max_entries=8)
def _picker_options(pairs: tuple[tuple[str, str], ...]) -> list[str]:
    """Cached so 30,490 option strings are not rebuilt on every rerun."""
    return sorted(f"{item}  @  {store}" for item, store in pairs)


def series_picker(frame: pd.DataFrame, label: str = "Item and store",
                  key: str | None = None, disabled: bool = False) -> tuple[str, str] | None:
    """Searchable item x store picker that stays usable at full-dataset scale.

    A raw selectbox over 30,490 series is unusable, so the list is filtered by a
    search box and capped, with the cap stated rather than silently applied.
    """
    if frame.empty:
        st.caption("No series available for the current filters.")
        return None
    pairs = tuple(
        frame[["item_id", "store_id"]].astype(str).drop_duplicates()
        .itertuples(index=False, name=None))
    options = _picker_options(pairs)
    total = len(options)

    if total > MAX_PICKER_OPTIONS:
        query = st.text_input(
            "Search items", key=f"{key}_search" if key else None,
            placeholder="e.g. FOODS_3_090 or CA_1", disabled=disabled,
            help="Type part of an item or store id to narrow the list.")
        if query:
            needle = query.strip().upper()
            options = [o for o in options if needle in o.upper()]
        shown = options[:MAX_PICKER_OPTIONS]
        if len(options) > MAX_PICKER_OPTIONS:
            st.caption(f"Showing the first {MAX_PICKER_OPTIONS:,} of "
                       f"{len(options):,} matching series - refine the search to "
                       f"reach the rest.")
    else:
        shown = options

    if not shown:
        st.caption("No series matches that search.")
        return None
    choice = st.selectbox(label, shown, key=key, disabled=disabled)
    if not choice:
        return None
    item_id, store_id = [part.strip() for part in choice.split("@")]
    return item_id, store_id


def fmt_units(value: float) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.0f}"


def fmt_money(value: float) -> str:
    if value is None or pd.isna(value):
        return "-"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}k"
    return f"${value:,.0f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value * 100:.{digits}f}%"


def fmt_metric(value: float, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"


METRIC_HELP = {
    "mae": "Mean absolute error in units, averaged over every item-day.",
    "rmse": "Root mean squared error in units. Large misses count more.",
    "rmsse": ("Root mean squared scaled error. Below 1.0 means the model beats a "
              "one-day naive forecast on that series' own training history. "
              "This is per-series RMSSE - the official M5 WRMSSE aggregate is "
              "deliberately not implemented, so it is not claimed."),
    "wape": ("Total absolute error divided by total actual units. The most reliable "
             "headline metric on intermittent demand."),
    "smape": ("Symmetric mean absolute percentage error. Unstable on intermittent "
              "demand: a naive forecast of exactly 0 scores perfectly on zero-sale "
              "days while any positive forecast is penalised 200%. Read WAPE first."),
    "bias": "Total forecast minus total actual, as a share of actual. Positive = over-forecasting.",
}
