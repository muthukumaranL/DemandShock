"""Config validation, artifact schemas, and app degradation behaviour.

These guard the interface between the pipeline and everything that consumes it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from demandshock import features as F  # noqa: E402
from demandshock.config import ConfigError, load_config  # noqa: E402


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def test_config_loads_and_exposes_resolved_paths(cfg):
    assert cfg.mode in ("development", "full")
    assert cfg.m5_dir.is_absolute()
    assert cfg.artifacts_dir.is_absolute()
    assert len(cfg.hash()) == 12


def test_missing_config_gives_an_actionable_error():
    with pytest.raises(ConfigError, match="not found"):
        load_config(REPO_ROOT / "definitely_not_here.yaml")


def test_invalid_mode_is_rejected():
    with pytest.raises(ConfigError, match="mode must be"):
        load_config(REPO_ROOT / "config.yaml", mode="sideways")


def test_shock_weights_must_sum_to_one_hundred(tmp_path):
    import yaml

    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["shock"]["weights"]["std_residual"] = 99
    bad = tmp_path / "config.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="must sum to 100"):
        load_config(bad)


def test_folds_must_validate_after_training(tmp_path):
    import yaml

    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["folds"]["F1"]["val_start_d"] = raw["folds"]["F1"]["train_end_d"] - 5
    bad = tmp_path / "config.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="chronological"):
        load_config(bad)


def test_no_windows_paths_are_hardcoded_in_source():
    offenders = []
    for path in list(REPO_ROOT.glob("src/**/*.py")) + list(REPO_ROOT.glob("app/**/*.py")) \
            + list(REPO_ROOT.glob("api/**/*.py")) + list(REPO_ROOT.glob("scripts/**/*.py")):
        text = path.read_text(encoding="utf-8")
        for marker in ("C:\\\\", "E:\\\\", "D:\\\\", "/mnt/c/"):
            if marker in text:
                offenders.append(f"{path.name} contains {marker}")
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# artifact schemas
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def built(cfg) -> bool:
    return (cfg.artifacts_dir / "model_metadata.json").exists()


REQUIRED_COLUMNS = {
    "metrics.parquet": {"model", "config", "fold_id", "horizon", "level", "group_key",
                        "mae", "rmse", "rmsse", "wape", "smape", "bias", "n_series"},
    "ablation.parquet": {"config", "fold_id", "horizon", "wape", "rmsse", "bias",
                         "n_features", "train_rows", "best_iteration"},
    "shocks_daily.parquet": {"item_id", "store_id", "d", "date", "residual", "z",
                             "score", "band", "guard_mult", "warmup", "sq_z",
                             "sq_pct", "sq_pers", "sq_vol", "sq_level"},
    "shock_episodes.parquet": {"episode_id", "item_id", "store_id", "start_date",
                               "end_date", "n_days", "peak_score", "band",
                               "classification", "why_flagged", "fema_context",
                               "econ_context", "fema_overlap"},
    "inventory_base.parquet": {"item_id", "store_id", "total_forecast_28",
                               "sigma_daily_resid", "latest_sell_price",
                               "under_exposure_usd", "over_exposure_usd", "eligible"},
    "feature_importance.parquet": {"feature", "family", "gain", "split",
                                   "mean_abs_contrib", "rank"},
    "residual_quantiles.parquet": {"store_id", "cat_id", "bucket", "q10", "q50", "q90"},
}


@pytest.mark.parametrize("filename,columns", sorted(REQUIRED_COLUMNS.items()))
def test_artifact_has_its_contracted_columns(cfg, built, filename, columns):
    if not built:
        pytest.skip("artifacts not built")
    path = cfg.artifacts_dir / filename
    if not path.exists():
        pytest.skip(f"{filename} not produced in this run")
    actual = set(pd.read_parquet(path).columns)
    assert columns.issubset(actual), f"{filename} is missing {columns - actual}"


def test_model_metadata_records_full_provenance(cfg, built):
    if not built:
        pytest.skip("artifacts not built")
    md = json.loads((cfg.artifacts_dir / "model_metadata.json").read_text("utf-8"))
    for key in ("model_name", "model_version", "mode", "trained_at", "config_hash",
                "selected_config", "shock_config", "features", "folds",
                "library_versions", "train_window_days", "training_rows_deploy"):
        assert key in md, f"model_metadata.json is missing {key}"
    assert md["n_features"] == len(md["features"])
    assert md["selected_config"] in ("A", "B", "C", "D")


def test_forecast_store_is_partitioned_and_complete(cfg, built):
    if not built:
        pytest.skip("artifacts not built")
    root = cfg.artifacts_dir / "forecasts.parquet"
    assert root.exists()
    md = json.loads((cfg.artifacts_dir / "model_metadata.json").read_text("utf-8"))
    selected = md["selected_config"]
    for fold in ("F1", "F2", "F3", "HOLDOUT", "FORWARD"):
        part = root / "model=lgbm" / f"config={selected}" / f"fold_id={fold}"
        assert part.exists(), f"missing forecast partition for {fold}"
    holdout = pd.read_parquet(root / "model=lgbm" / f"config={selected}"
                              / "fold_id=HOLDOUT")
    assert {"p10", "p90"}.issubset(holdout.columns), "holdout must carry intervals"
    assert (holdout["y_pred"] >= 0).all(), "forecasts must be non-negative"
    assert (holdout["p10"] <= holdout["p90"]).all(), "interval bounds inverted"
    assert set(holdout["step"]) == set(range(1, 29))


def test_forward_forecast_carries_no_actuals(cfg, built):
    if not built:
        pytest.skip("artifacts not built")
    md = json.loads((cfg.artifacts_dir / "model_metadata.json").read_text("utf-8"))
    part = (cfg.artifacts_dir / "forecasts.parquet" / "model=lgbm"
            / f"config={md['selected_config']}" / "fold_id=FORWARD")
    forward = pd.read_parquet(part)
    assert forward["y_true"].isna().all(), (
        "the forward window is beyond the data; it cannot have actuals")


def test_ablation_arms_are_cumulative_in_feature_count(cfg, built):
    if not built:
        pytest.skip("artifacts not built")
    path = cfg.artifacts_dir / "ablation.parquet"
    if not path.exists():
        pytest.skip("no ablation in this run")
    ablation = pd.read_parquet(path)
    counts = ablation.groupby("config")["n_features"].first()
    for smaller, bigger in (("A", "B"), ("B", "C"), ("C", "D")):
        if smaller in counts and bigger in counts:
            assert counts[smaller] < counts[bigger]


def test_shock_scores_are_bounded_and_banded(cfg, built):
    if not built:
        pytest.skip("artifacts not built")
    path = cfg.artifacts_dir / "shocks_daily.parquet"
    if not path.exists():
        pytest.skip("no shock artifacts")
    daily = pd.read_parquet(path)
    assert daily["score"].between(0, 100.001).all()
    assert set(daily["band"]).issubset(
        {"Normal", "Watch", "Elevated", "Severe", "Critical"})


def test_data_quality_report_has_no_failures(cfg, built):
    if not built:
        pytest.skip("artifacts not built")
    quality = json.loads((cfg.artifacts_dir / "data_quality.json").read_text("utf-8"))
    failures = [c["name"] for c in quality["checks"] if c["status"] == "fail"]
    assert not failures, f"data-quality failures present: {failures}"


# ---------------------------------------------------------------------------
# documentation cannot drift from measurements
# ---------------------------------------------------------------------------
def test_results_are_generated_not_handwritten():
    results = REPO_ROOT / "RESULTS.md"
    if not results.exists():
        pytest.skip("RESULTS.md not generated yet")
    text = results.read_text(encoding="utf-8")
    assert "scripts/export_results.py" in text
    assert "none is typed by hand" in text


def test_readme_does_not_quote_hardcoded_metric_values():
    """Numbers belong in RESULTS.md, which is generated from artifacts."""
    import re

    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    # a WAPE/RMSSE claim with a literal decimal number would be a hand-typed metric
    offenders = re.findall(
        r"(?:WAPE|RMSSE|sMAPE|accuracy)\s*(?:of|=|:|was|is)?\s*(\d+\.\d+)", text,
        flags=re.IGNORECASE)
    assert not offenders, (
        f"README quotes literal metric values {offenders}; keep measured numbers in "
        "RESULTS.md so they cannot go stale")


# ---------------------------------------------------------------------------
# app degradation
# ---------------------------------------------------------------------------
APP_PAGES = ["app/Home.py"] + sorted(
    str(p.relative_to(REPO_ROOT)).replace("\\", "/")
    for p in (REPO_ROOT / "app" / "pages").glob("*.py"))


@pytest.mark.parametrize("page", APP_PAGES)
def test_every_page_renders_without_error(cfg, built, page):
    if not built:
        pytest.skip("artifacts not built")
    from streamlit.testing.v1 import AppTest

    sys.path.insert(0, str(REPO_ROOT / "app"))
    app = AppTest.from_file(str(REPO_ROOT / page), default_timeout=180).run()
    assert not app.exception, (
        f"{page} raised: {[str(e.value)[:300] for e in app.exception]}")


def test_pages_degrade_gracefully_when_artifacts_are_absent(tmp_path, monkeypatch):
    """An empty artifacts directory must produce guidance, never a traceback."""
    import yaml
    from streamlit.testing.v1 import AppTest

    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["paths"]["artifacts_dir"] = str((tmp_path / "empty_artifacts").as_posix())
    raw["paths"]["processed_dir"] = str((tmp_path / "empty_processed").as_posix())
    (tmp_path / "empty_artifacts").mkdir()
    (tmp_path / "empty_processed").mkdir()
    stub = tmp_path / "config.yaml"
    stub.write_text(yaml.safe_dump(raw), encoding="utf-8")

    sys.path.insert(0, str(REPO_ROOT / "app"))
    import shared

    monkeypatch.setattr(shared, "get_config", lambda: load_config(stub))
    shared.get_config.clear() if hasattr(shared.get_config, "clear") else None

    for page in APP_PAGES:
        app = AppTest.from_file(str(REPO_ROOT / page), default_timeout=120)
        app.run()
        assert not app.exception, (
            f"{page} raised a traceback with no artifacts instead of showing guidance: "
            f"{[str(e.value)[:200] for e in app.exception]}")


def test_forecast_loader_preserves_prediction_intervals(cfg, built):
    """Regression: the P10-P90 band silently vanished from every chart.

    Backtest folds carry no interval columns while the holdout does. PyArrow infers
    a dataset's schema from its FIRST fragment, so reading the partitioned forecast
    store dropped p10/p50/p90 everywhere - the band never rendered and nothing
    failed. The loader must unify schemas across fragments.
    """
    if not built:
        pytest.skip("artifacts not built")
    import sys
    sys.path.insert(0, str(REPO_ROOT / "app"))
    import shared as sh

    md = json.loads((cfg.artifacts_dir / "model_metadata.json").read_text("utf-8"))
    selected = md["selected_config"]

    holdout = sh.load_forecasts(model="lgbm", config=selected, fold="HOLDOUT")
    assert not holdout.empty
    for col in ("p10", "p90"):
        assert col in holdout.columns, (
            f"{col} missing from the holdout slice - the interval band cannot render")
        assert holdout[col].notna().all(), f"{col} came back null"
    assert (holdout["p10"] <= holdout["p90"]).all()

    # A fold that genuinely has no intervals must not gain phantom all-null ones.
    fold_one = sh.load_forecasts(model="lgbm", config=selected, fold="F1")
    if not fold_one.empty:
        assert "p10" not in fold_one.columns, (
            "folds without intervals should drop the column, not carry it as nulls")


# ---------------------------------------------------------------------------
# deployment bundle
# ---------------------------------------------------------------------------
def test_deploy_config_exists_and_writes_outside_the_local_tree():
    """The committed bundle must not collide with a local full-scale build."""
    deploy = REPO_ROOT / "config.deploy.yaml"
    if not deploy.exists():
        pytest.skip("no deployment config in this checkout")
    cfg = load_config(deploy)
    assert cfg.mode == "development", "the committed bundle must be the small build"
    for key in ("artifacts_dir", "processed_dir", "models_dir"):
        assert "deploy" in str(cfg.path_of(key)).replace("\\", "/"), (
            f"{key} must live under deploy/ so a local run cannot overwrite it")


def test_app_falls_back_to_the_deploy_bundle_when_nothing_is_built(tmp_path):
    """A fresh clone has no local artifacts; the app must still find data.

    This is the deployment path: without the fallback the hosted app renders its
    'artifacts not built' panel forever, which is exactly what a reviewer sees.
    """
    import sys as _sys
    import yaml

    _sys.path.insert(0, str(REPO_ROOT / "app"))
    import shared

    deploy = REPO_ROOT / "config.deploy.yaml"
    if not deploy.exists() or not (load_config(deploy).artifacts_dir
                                   / "model_metadata.json").exists():
        pytest.skip("deployment bundle not built in this checkout")

    # a config whose artifacts directory is empty, standing in for a fresh clone
    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    empty = tmp_path / "empty_artifacts"
    empty.mkdir()
    raw["paths"]["artifacts_dir"] = empty.as_posix()
    unbuilt = tmp_path / "config.yaml"
    unbuilt.write_text(yaml.safe_dump(raw), encoding="utf-8")

    chosen = shared.resolve_config(unbuilt, deploy)
    assert (chosen.artifacts_dir / "model_metadata.json").exists(), (
        "fell through to a config with no artifacts - the hosted app would be blank")
    assert chosen.mode == "development"

    # and when the primary IS built it must win, so local work is never shadowed
    primary = load_config(REPO_ROOT / "config.yaml")
    if (primary.artifacts_dir / "model_metadata.json").exists():
        assert shared.resolve_config(REPO_ROOT / "config.yaml",
                                     deploy).artifacts_dir == primary.artifacts_dir


def test_deploy_bundle_carries_every_artifact_the_app_reads():
    deploy = REPO_ROOT / "config.deploy.yaml"
    if not deploy.exists():
        pytest.skip("no deployment config in this checkout")
    cfg = load_config(deploy)
    if not (cfg.artifacts_dir / "model_metadata.json").exists():
        pytest.skip("deployment bundle not built in this checkout")
    for name in ("metrics.parquet", "ablation.parquet", "shocks_daily.parquet",
                 "shock_episodes.parquet", "inventory_base.parquet",
                 "feature_importance.parquet", "residual_quantiles.parquet",
                 "model_metadata.json", "data_quality.json", "forecasts.parquet"):
        assert (cfg.artifacts_dir / name).exists(), f"bundle is missing {name}"
    for name in ("calendar.parquet", "series_meta.parquet", "prices_long.parquet",
                 "fema_context.parquet", "fred_state_daily.parquet", "sales_long"):
        assert (cfg.processed_dir / name).exists(), f"bundle is missing {name}"
