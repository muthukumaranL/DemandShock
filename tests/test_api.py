"""API contract tests against the real built artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import api.main as api_main  # noqa: E402
from api.main import app, missing_artifacts  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def built() -> bool:
    return not missing_artifacts()


@pytest.fixture(scope="module")
def known_series(cfg, built):
    if not built:
        pytest.skip("artifacts not built")
    base = pd.read_parquet(cfg.artifacts_dir / "inventory_base.parquet")
    row = base[base["eligible"]].iloc[0]
    return str(row["item_id"]), str(row["store_id"])


def test_health_always_answers(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert "rebuild_command" in body


def test_metadata_reports_the_real_model(client, built):
    if not built:
        pytest.skip("artifacts not built")
    body = client.get("/metadata").json()
    assert body["selected_config"] in ("A", "B", "C", "D")
    assert body["n_features"] == len(body["features"])
    assert "HOLDOUT" in body["folds"]
    assert any("no inventory" in limit.lower() for limit in body["limitations"])


def test_forecast_returns_a_full_28_day_path(client, known_series):
    item_id, store_id = known_series
    body = client.get("/forecast", params={"item_id": item_id, "store_id": store_id}).json()
    assert body["has_actuals"] is True
    assert len(body["points"]) == 28
    assert [p["step"] for p in body["points"]] == list(range(1, 29))
    assert all(p["y_pred"] >= 0 for p in body["points"])


def test_forward_forecast_has_no_actuals(client, known_series):
    item_id, store_id = known_series
    response = client.get("/forecast", params={"item_id": item_id, "store_id": store_id,
                                               "fold": "FORWARD"})
    if response.status_code == 404:
        pytest.skip("forward forecasts not stored")
    body = response.json()
    assert body["has_actuals"] is False
    assert all(p["y_true"] is None for p in body["points"])


def test_unknown_series_is_a_clean_404(client, built):
    if not built:
        pytest.skip("artifacts not built")
    response = client.get("/forecast", params={"item_id": "NOT_A_REAL_ITEM",
                                               "store_id": "CA_1"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "series_not_found"


def test_shocks_filter_and_never_claim_causation(client, built):
    if not built:
        pytest.skip("artifacts not built")
    body = client.get("/shocks", params={"min_score": 70, "limit": 5}).json()
    assert body["count"] <= 5
    assert "never causation" in body["interpretation"]
    for episode in body["episodes"]:
        assert episode["peak_score"] >= 70
        assert "components" in episode["why_flagged"]


def test_shocks_empty_result_is_200_not_an_error(client, built):
    if not built:
        pytest.skip("artifacts not built")
    response = client.get("/shocks", params={"store_id": "NO_SUCH_STORE"})
    assert response.status_code == 200
    assert response.json()["count"] == 0


def test_metrics_endpoint_disclaims_wrmsse(client, built):
    if not built:
        pytest.skip("artifacts not built")
    body = client.get("/metrics", params={"level": "overall", "horizon": 28}).json()
    assert body["count"] > 0
    assert "wrmsse is not implemented" in body["note"].lower()


def test_inventory_analysis_returns_transparent_arithmetic(client, known_series):
    item_id, store_id = known_series
    response = client.post("/inventory-analysis", json={
        "item_id": item_id, "store_id": store_id,
        "lead_time_days": 7, "service_level": 0.95, "on_hand_units": 25.0})
    assert response.status_code == 200
    body = response.json()
    assert body["inputs_are_user_supplied"] is True
    assert body["z"] == pytest.approx(1.6448536, abs=1e-5)
    assert body["reorder_point"] == pytest.approx(
        body["lead_time_demand"] + body["safety_stock"], abs=1e-4)
    assert body["safety_stock"] == pytest.approx(
        body["z"] * body["sigma_lead_time"], abs=1e-4)
    assert any("no inventory records" in a.lower() for a in body["assumptions"])


def test_inventory_validation_rejects_impossible_inputs(client, known_series):
    item_id, store_id = known_series
    for payload in (
        {"lead_time_days": 0, "service_level": 0.95},
        {"lead_time_days": 7, "service_level": 1.2},
        {"lead_time_days": 400, "service_level": 0.95},
    ):
        response = client.post("/inventory-analysis",
                               json={"item_id": item_id, "store_id": store_id, **payload})
        assert response.status_code == 422


def test_schema_validation_uses_the_documented_error_envelope(client, known_series):
    """A pydantic failure must look like every other error, not FastAPI's default."""
    item_id, store_id = known_series
    response = client.post("/inventory-analysis", json={
        "item_id": item_id, "store_id": store_id,
        "lead_time_days": 7, "service_level": 5.0})
    assert response.status_code == 422
    body = response.json()
    assert "error" in body, f"expected the {{'error': ...}} envelope, got {body}"
    assert body["error"]["code"] == "validation_error"
    assert "service_level" in body["error"]["message"]


def test_a_defaulted_service_level_is_declared_not_presented_as_user_input(
        client, known_series):
    """M5 has no service levels; if the API supplies one, it must say so."""
    item_id, store_id = known_series
    body = client.post("/inventory-analysis", json={
        "item_id": item_id, "store_id": store_id, "lead_time_days": 7}).json()
    assert body["service_level_defaulted"] is True
    assert body["inputs_are_user_supplied"] is False
    assert body["service_level"] == pytest.approx(0.95)
    assert any("default" in a.lower() for a in body["assumptions"])

    supplied = client.post("/inventory-analysis", json={
        "item_id": item_id, "store_id": store_id,
        "lead_time_days": 7, "service_level": 0.99}).json()
    assert supplied["service_level_defaulted"] is False
    assert supplied["inputs_are_user_supplied"] is True
    assert supplied["service_level"] == pytest.approx(0.99)


def test_missing_artifacts_return_503_with_a_rebuild_command(client, monkeypatch, built):
    if not built:
        pytest.skip("artifacts not built")
    monkeypatch.setitem(api_main.REQUIRED, "metrics",
                        "artifacts/__does_not_exist__.parquet")
    response = client.get("/metrics")
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "artifacts_missing"
    assert "run_pipeline" in error["message"]


def test_list_endpoints_disclose_truncation(client, built):
    if not built:
        pytest.skip("artifacts not built")
    shocks = client.get("/shocks", params={"min_score": 0, "limit": 1}).json()
    assert shocks["count"] == 1
    assert shocks["total_matching"] >= shocks["count"]
    assert shocks["truncated"] == (shocks["total_matching"] > shocks["count"])

    metrics = client.get("/metrics", params={"limit": 2}).json()
    assert metrics["count"] <= 2
    assert metrics["total_matching"] >= metrics["count"]


def test_metrics_truncation_is_deterministic(client, built):
    if not built:
        pytest.skip("artifacts not built")
    first = client.get("/metrics", params={"limit": 5}).json()["rows"]
    second = client.get("/metrics", params={"limit": 5}).json()["rows"]
    assert first == second, "the same query must drop the same rows every time"


def test_openapi_schema_builds(client):
    assert client.get("/openapi.json").status_code == 200
