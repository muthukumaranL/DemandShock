"""Minimal read-only inference/data service over the same artifacts the app uses.

Deliberately small: six endpoints, no auth, no write paths, no training. It exists
so forecasts, shocks and planning arithmetic can be consumed programmatically
without going through Streamlit.

Run:  uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from demandshock import inventory as INV  # noqa: E402
from demandshock.config import load_config  # noqa: E402

REBUILD = "python scripts/run_pipeline.py --mode development"

app = FastAPI(
    title="DemandShock API",
    version="1.0.0",
    description=("Read-only access to DemandShock forecasts, demand-shock episodes, "
                 "accuracy metrics and inventory planning arithmetic. All figures "
                 "derive from real M5, FEMA and FRED data."),
)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class ErrorBody(BaseModel):
    code: str
    message: str


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    artifacts_loaded: bool
    mode: str | None = None
    missing: list[str] = Field(default_factory=list)
    rebuild_command: str = REBUILD


class Metadata(BaseModel):
    # `model_*` fields are meaningful here (they describe the forecasting model),
    # so opt out of pydantic's protected namespace instead of renaming them.
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    mode: str
    trained_at: str
    strategy: str
    selected_config: str
    selected_config_label: str
    shock_config: str
    objective: str
    n_features: int
    features: list[str]
    train_window_days: int | None
    series_count: int
    folds: dict[str, Any]
    horizons: list[int]
    available_models: list[str]
    available_configs: list[str]
    library_versions: dict[str, str]
    limitations: list[str]


class ForecastPoint(BaseModel):
    date: date
    d: int
    step: int
    y_pred: float
    y_true: float | None = None
    p10: float | None = None
    p90: float | None = None


class ForecastResponse(BaseModel):
    item_id: str
    store_id: str
    product: str | None = Field(
        default=None,
        description="Measured description of the series (department, price band "
                    "within its category, sales velocity). M5 anonymises product "
                    "names, so this is derived from real attributes, not a name.")
    model: str
    config: str
    fold: str
    has_actuals: bool
    points: list[ForecastPoint]


class Episode(BaseModel):
    episode_id: str
    item_id: str
    store_id: str
    product: str | None = None
    state_id: str | None
    start_date: date
    end_date: date
    n_days: int
    peak_score: float
    band: str
    classification: str
    fema_overlap: bool
    why_flagged: dict[str, Any]
    fema_context: dict[str, Any]
    econ_context: dict[str, Any]


class ShockList(BaseModel):
    count: int
    total_matching: int
    truncated: bool
    scored_window: dict[str, str]
    interpretation: str
    episodes: list[Episode]


class MetricRow(BaseModel):
    model: str
    config: str
    fold_id: str
    horizon: int | None
    level: str
    group_key: str
    mae: float | None
    rmse: float | None
    rmsse: float | None
    wape: float | None
    smape: float | None
    bias: float | None
    n_series: int | None


class MetricsResponse(BaseModel):
    count: int
    total_matching: int
    truncated: bool
    note: str
    rows: list[MetricRow]


class InventoryRequest(BaseModel):
    item_id: str
    store_id: str
    lead_time_days: int = Field(ge=1, le=28,
                                description="User-supplied. M5 contains no lead times.")
    service_level: float = Field(default=0.95, gt=0.5, lt=1.0,
                                 description="User-supplied operating target.")
    on_hand_units: float | None = Field(default=None, ge=0)
    unit_cost: float | None = Field(default=None, ge=0)


class InventoryResponse(BaseModel):
    item_id: str
    store_id: str
    product: str | None = None
    inputs_are_user_supplied: bool
    service_level: float
    service_level_defaulted: bool
    z: float
    sigma_daily: float
    lead_time_days_used: int
    lead_time_clamped: bool
    lead_time_demand: float
    sigma_lead_time: float
    sigma_lead_time_conservative: float
    safety_stock: float
    reorder_point: float
    days_of_cover: float | None
    projected_stockout_day: int | None
    uncovered_units: float | None
    formulas: dict[str, str]
    assumptions: list[str]


# ---------------------------------------------------------------------------
# artifact access
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_cfg():
    return load_config(REPO_ROOT / "config.yaml")


REQUIRED = {
    "model_metadata": "artifacts/model_metadata.json",
    "metrics": "artifacts/metrics.parquet",
    "shock_episodes": "artifacts/shock_episodes.parquet",
    "inventory_base": "artifacts/inventory_base.parquet",
    "forecasts": "artifacts/forecasts.parquet",
    "calendar": "data/processed/calendar.parquet",
    "series_meta": "data/processed/series_meta.parquet",
}


def missing_artifacts() -> list[str]:
    return [name for name, rel in REQUIRED.items() if not (REPO_ROOT / rel).exists()]


def _require(*names: str) -> None:
    absent = [n for n in names if n in missing_artifacts()]
    if absent:
        raise HTTPException(
            status_code=503,
            detail={"code": "artifacts_missing",
                    "message": f"Missing artifacts: {', '.join(absent)}. "
                               f"Build them with: {REBUILD}"})


def _mtime(path: Path) -> float:
    if path.is_dir():
        return max((p.stat().st_mtime for p in path.rglob("*.parquet")), default=0.0)
    return path.stat().st_mtime


@lru_cache(maxsize=32)
def _load_parquet(rel: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(REPO_ROOT / rel)


def load(name: str) -> pd.DataFrame:
    rel = REQUIRED[name]
    return _load_parquet(rel, _mtime(REPO_ROOT / rel))


@lru_cache(maxsize=4)
def _load_json(rel: str, mtime: float) -> dict:
    return json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))


def metadata() -> dict:
    rel = REQUIRED["model_metadata"]
    return _load_json(rel, _mtime(REPO_ROOT / rel))


def day_to_date() -> dict[int, pd.Timestamp]:
    cal = load("calendar")
    return dict(zip(cal["d"], cal["date"]))


def product_labels() -> dict[tuple[str, str], str]:
    """(item, store) -> measured descriptor.

    M5 anonymises product identities, so instead of a name each series carries a
    description derived from its own data: department, price band within its
    category, and sales velocity. Returns an empty map for artifact sets built
    before descriptors existed, so responses degrade to bare ids rather than fail.
    """
    try:
        meta = load("series_meta")
    except (KeyError, FileNotFoundError):
        return {}
    if "dept_label" not in meta.columns:
        return {}
    return {
        (str(r.item_id), str(r.store_id)):
            f"{r.dept_label} · {r.price_band} · {r.velocity}"
            + (f" (${r.median_price:.2f})" if pd.notna(r.median_price) else "")
        for r in meta.itertuples(index=False)
    }


LIMITATIONS = [
    "M5 anonymises product identities: the source data contains no product names. "
    "The 'product' field is a description measured from each series' own history "
    "(department, price band within its category, sales velocity), not a name.",
    "M5 contains no inventory records; lead time, service level, on-hand units and "
    "unit cost are user-supplied operating assumptions.",
    "FEMA declarations are county-scoped while M5 discloses only a store's state, so "
    "disaster information is a state-level temporal coincidence, never a cause.",
    "FRED unemployment is monthly and joined with a publication lag; it cannot "
    "explain a daily deviation.",
    "Shock detection is retrospective: residuals exist only where the model forecast "
    "out of sample.",
    "The official M5 WRMSSE aggregate is not implemented and is never reported.",
    "The demand shock score is a project-defined metric, not an industry standard.",
]


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=Health, tags=["service"])
def health() -> Health:
    absent = missing_artifacts()
    mode = None
    if "model_metadata" not in absent:
        mode = metadata().get("mode")
    return Health(status="ok" if not absent else "degraded",
                  artifacts_loaded=not absent, mode=mode, missing=absent)


@app.get("/metadata", response_model=Metadata, tags=["service"])
def get_metadata() -> Metadata:
    _require("model_metadata", "forecasts")
    md = metadata()
    root = REPO_ROOT / REQUIRED["forecasts"]
    models = sorted({p.name.split("=", 1)[1] for p in root.glob("model=*")})
    configs = sorted({p.name.split("=", 1)[1]
                      for p in root.glob("model=*/config=*") if "na" not in p.name})
    return Metadata(
        model_name=md["model_name"], model_version=md["model_version"],
        mode=md["mode"], trained_at=md["trained_at"], strategy=md["strategy"],
        selected_config=md["selected_config"],
        selected_config_label=md.get("selected_config_label", ""),
        shock_config=md["shock_config"], objective=md["objective"],
        n_features=md["n_features"], features=md["features"],
        train_window_days=md.get("train_window_days"),
        series_count=md["series_count"], folds=md["folds"],
        horizons=list(get_cfg()["horizons"]),
        available_models=models, available_configs=configs,
        library_versions=md.get("library_versions", {}),
        limitations=LIMITATIONS)


@app.get("/forecast", response_model=ForecastResponse, tags=["forecast"])
def get_forecast(
    item_id: str = Query(..., description="M5 item id, e.g. FOODS_1_001"),
    store_id: str = Query(..., description="M5 store id, e.g. CA_1"),
    model: str = Query("lgbm"),
    config: str | None = Query(None, description="Feature set; defaults to the "
                                                 "selected one"),
    fold: str = Query("HOLDOUT", description="F1, F2, F3, HOLDOUT or FORWARD"),
) -> ForecastResponse:
    _require("forecasts", "model_metadata", "calendar")
    config = config or (metadata()["selected_config"] if model == "lgbm" else "na")
    part = (REPO_ROOT / REQUIRED["forecasts"] / f"model={model}"
            / f"config={config}" / f"fold_id={fold}")
    if not part.exists():
        raise HTTPException(
            status_code=404,
            detail={"code": "forecast_not_found",
                    "message": f"No stored forecasts for model={model}, "
                               f"config={config}, fold={fold}."})
    frame = _load_parquet(str(part.relative_to(REPO_ROOT)).replace("\\", "/"),
                          _mtime(part))
    frame = frame[(frame["item_id"].astype(str) == item_id)
                  & (frame["store_id"].astype(str) == store_id)].sort_values("d")
    if frame.empty:
        raise HTTPException(
            status_code=404,
            detail={"code": "series_not_found",
                    "message": f"No forecast for {item_id} at {store_id} in {fold}."})

    dates = day_to_date()
    points = []
    for row in frame.itertuples(index=False):
        y_true = getattr(row, "y_true", None)
        points.append(ForecastPoint(
            date=dates[int(row.d)].date(), d=int(row.d), step=int(row.step),
            y_pred=round(float(row.y_pred), 4),
            y_true=None if y_true is None or pd.isna(y_true) else float(y_true),
            p10=round(float(row.p10), 4) if hasattr(row, "p10") and pd.notna(row.p10) else None,
            p90=round(float(row.p90), 4) if hasattr(row, "p90") and pd.notna(row.p90) else None))
    return ForecastResponse(
        item_id=item_id, store_id=store_id,
        product=product_labels().get((item_id, store_id)),
        model=model, config=config, fold=fold,
        has_actuals=any(p.y_true is not None for p in points), points=points)


@app.get("/shocks", response_model=ShockList, tags=["shocks"])
def get_shocks(
    store_id: str | None = None,
    state_id: str | None = None,
    classification: str | None = None,
    min_score: float = Query(30.0, ge=0, le=100),
    start: date | None = None,
    end: date | None = None,
    limit: int = Query(100, ge=1, le=1000),
) -> ShockList:
    _require("shock_episodes")
    frame = load("shock_episodes").copy()
    if store_id:
        frame = frame[frame["store_id"].astype(str) == store_id]
    if state_id:
        frame = frame[frame["state_id"].astype(str) == state_id]
    if classification:
        frame = frame[frame["classification"] == classification]
    frame = frame[frame["peak_score"] >= min_score]
    if start:
        frame = frame[frame["end_date"] >= pd.Timestamp(start)]
    if end:
        frame = frame[frame["start_date"] <= pd.Timestamp(end)]
    total_matching = int(len(frame))
    frame = frame.sort_values("peak_score", ascending=False).head(limit)

    labels = product_labels()
    episodes = [
        Episode(
            episode_id=row.episode_id, item_id=str(row.item_id),
            store_id=str(row.store_id),
            product=labels.get((str(row.item_id), str(row.store_id))),
            state_id=None if pd.isna(row.state_id) else str(row.state_id),
            start_date=pd.Timestamp(row.start_date).date(),
            end_date=pd.Timestamp(row.end_date).date(),
            n_days=int(row.n_days), peak_score=round(float(row.peak_score), 2),
            band=str(row.band), classification=str(row.classification),
            fema_overlap=bool(row.fema_overlap),
            why_flagged=json.loads(row.why_flagged),
            fema_context=json.loads(row.fema_context),
            econ_context=json.loads(row.econ_context))
        for row in frame.itertuples(index=False)
    ]
    md = metadata()
    folds = md.get("folds", {})
    window = {
        "start": min((f["val_start_date"] for k, f in folds.items() if k != "FORWARD"),
                     default="?"),
        "end": max((f["val_end_date"] for k, f in folds.items() if k != "FORWARD"),
                   default="?"),
    }
    return ShockList(
        count=len(episodes), total_matching=total_matching,
        truncated=total_matching > len(episodes), scored_window=window,
        interpretation=("Scores measure deviation from model expectation on days that "
                        "already happened. FEMA overlap indicates temporal coincidence "
                        "within a state, never causation."),
        episodes=episodes)


@app.get("/metrics", response_model=MetricsResponse, tags=["forecast"])
def get_metrics(
    model: str | None = None,
    config: str | None = None,
    fold_id: str | None = None,
    horizon: int | None = None,
    level: str = Query("overall", description="overall, state, store, category, "
                                              "department or day"),
    limit: int = Query(500, ge=1, le=5000),
) -> MetricsResponse:
    _require("metrics")
    frame = load("metrics")
    frame = frame[frame["level"] == level]
    for column, value in (("model", model), ("config", config), ("fold_id", fold_id)):
        if value is not None:
            frame = frame[frame[column] == value]
    if horizon is not None:
        frame = frame[frame["horizon"] == horizon]
    total_matching = int(len(frame))
    # Deterministic ordering before truncation, so `limit` drops a predictable tail
    # rather than an arbitrary one.
    frame = frame.sort_values(
        ["model", "config", "fold_id", "level", "horizon", "group_key"],
        kind="stable", na_position="last").head(limit)

    rows = [
        MetricRow(
            model=str(r.model), config=str(r.config), fold_id=str(r.fold_id),
            horizon=None if pd.isna(r.horizon) else int(r.horizon),
            level=str(r.level), group_key=str(r.group_key),
            **{m: (None if pd.isna(getattr(r, m)) else round(float(getattr(r, m)), 6))
               for m in ("mae", "rmse", "rmsse", "wape", "smape", "bias")},
            n_series=None if pd.isna(r.n_series) else int(r.n_series))
        for r in frame.itertuples(index=False)
    ]
    return MetricsResponse(
        count=len(rows), total_matching=total_matching,
        truncated=total_matching > len(rows),
        note=("RMSSE is per-series and aggregated as an unweighted mean; the official "
              "M5 WRMSSE is not implemented and is not reported. sMAPE is unstable on "
              "intermittent demand - prefer WAPE."),
        rows=rows)


@app.post("/inventory-analysis", response_model=InventoryResponse, tags=["inventory"])
def inventory_analysis(request: InventoryRequest) -> InventoryResponse:
    _require("inventory_base", "forecasts", "model_metadata")
    base = load("inventory_base")
    row = base[(base["item_id"].astype(str) == request.item_id)
               & (base["store_id"].astype(str) == request.store_id)]
    if row.empty:
        raise HTTPException(
            status_code=404,
            detail={"code": "series_not_found",
                    "message": f"{request.item_id} at {request.store_id} is not in "
                               f"the inventory base table."})
    row = row.iloc[0]
    if not bool(row["eligible"]):
        raise HTTPException(
            status_code=422,
            detail={"code": "insufficient_history",
                    "message": "This series has too little price or residual history "
                               "for planning arithmetic."})

    md = metadata()
    part = (REPO_ROOT / REQUIRED["forecasts"] / "model=lgbm"
            / f"config={md['selected_config']}" / "fold_id=HOLDOUT")
    frame = _load_parquet(str(part.relative_to(REPO_ROOT)).replace("\\", "/"),
                          _mtime(part))
    path = INV.daily_forecast_path(frame, request.item_id, request.store_id)
    if path.empty:
        raise HTTPException(
            status_code=404,
            detail={"code": "forecast_not_found",
                    "message": "No forecast path stored for this series."})

    result = INV.plan(
        get_cfg(), path["y_pred"].to_numpy(), float(row["sigma_daily_resid"]),
        request.lead_time_days, request.service_level,
        on_hand_units=request.on_hand_units, unit_cost=request.unit_cost,
        unit_price=float(row["latest_sell_price"]))

    # A defaulted service level is an assumption WE made, not one the caller
    # supplied - say so rather than asserting every input was user-provided.
    defaulted = "service_level" not in request.model_fields_set
    assumptions = list(result["assumptions"])
    if defaulted:
        assumptions.insert(0, (
            f"service_level {request.service_level} was applied by default because "
            f"the request omitted it - supply your own operating target."))

    return InventoryResponse(
        item_id=request.item_id, store_id=request.store_id,
        product=product_labels().get((request.item_id, request.store_id)),
        inputs_are_user_supplied=not defaulted,
        service_level=float(result["service_level"]),
        service_level_defaulted=defaulted,
        z=round(result["z"], 6), sigma_daily=round(result["sigma_daily"], 6),
        lead_time_days_used=result["lead_time_days_used"],
        lead_time_clamped=result["lead_time_clamped"],
        lead_time_demand=round(result["lead_time_demand"], 4),
        sigma_lead_time=round(result["sigma_lead_time"], 4),
        sigma_lead_time_conservative=round(result["sigma_lead_time_conservative"], 4),
        safety_stock=round(result["safety_stock"], 4),
        reorder_point=round(result["reorder_point"], 4),
        days_of_cover=(None if result["days_of_cover"] is None
                       else round(result["days_of_cover"], 3)),
        projected_stockout_day=result["projected_stockout_day"],
        uncovered_units=(None if result["uncovered_units"] is None
                         else round(result["uncovered_units"], 4)),
        formulas=result["formulas"], assumptions=assumptions)


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request, exc: RequestValidationError):
    """Schema failures use the same error envelope as everything else."""
    fields = ", ".join(".".join(str(p) for p in e.get("loc", ())[1:]) or "body"
                       for e in exc.errors()) or "request"
    return JSONResponse(
        status_code=422,
        content={"error": {
            "code": "validation_error",
            "message": f"Invalid request parameters: {fields}.",
            "detail": jsonable_encoder(exc.errors()),
        }})


@app.exception_handler(HTTPException)
def http_exception_handler(request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail:
        body = ErrorBody(code=detail["code"], message=detail["message"])
    else:
        body = ErrorBody(code="error", message=str(detail))
    return JSONResponse(status_code=exc.status_code, content={"error": body.model_dump()})
