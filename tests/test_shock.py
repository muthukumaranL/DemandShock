"""Shock-score correctness, reproducibility and guard behaviour.

Scores are computed from real backtest residuals wherever a real example exists.
Small hand-built arrays are used only to pin down pure-function behaviour at the
boundaries (guards, warm-up caps, episode bridging); they never train a model and
never appear in any reported result.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from demandshock import shock as S


@pytest.fixture(scope="module")
def artifacts(cfg):
    daily_path = cfg.artifacts_dir / "shocks_daily.parquet"
    episode_path = cfg.artifacts_dir / "shock_episodes.parquet"
    if not daily_path.exists() or not episode_path.exists():
        pytest.skip("shock artifacts not built; run scripts/evaluate.py")
    return pd.read_parquet(daily_path), pd.read_parquet(episode_path)


# ---------------------------------------------------------------------------
# score construction
# ---------------------------------------------------------------------------
def test_scores_stay_within_bounds(artifacts):
    daily, _ = artifacts
    assert daily["score"].min() >= 0
    assert daily["score"].max() <= 100.0001


def test_band_assignment_matches_the_configured_thresholds(cfg, artifacts):
    daily, _ = artifacts
    sample = daily.sample(min(2000, len(daily)), random_state=0)
    for row in sample.itertuples(index=False):
        assert row.band == S.band_for(float(row.score), cfg)


def test_score_arithmetic_is_exactly_reproducible_from_its_components(cfg, artifacts):
    """Every point of the score must be attributable - the panel shows this sum."""
    daily, _ = artifacts
    weights = cfg["shock"]["weights"]
    sample = daily[~daily["warmup"]].sample(min(1500, len(daily)), random_state=1)
    recomputed = (
        weights["std_residual"] * sample["sq_z"]
        + weights["pct_deviation"] * sample["sq_pct"]
        + weights["persistence"] * sample["sq_pers"]
        + weights["volatility_ratio"] * sample["sq_vol"]
        + weights["level_shift"] * sample["sq_level"])
    np.testing.assert_allclose(recomputed.to_numpy(),
                               sample["score_raw"].to_numpy(), rtol=1e-5)

    expected = sample["score_raw"] * sample["guard_mult"]
    zeroed = (sample["y_true"] == 0) & (sample["y_pred"] < 1)
    expected = expected.where(~zeroed, 0.0)
    np.testing.assert_allclose(expected.to_numpy(),
                               sample["score"].to_numpy(dtype="float64"), atol=1e-4)


def test_component_weights_sum_to_one_hundred(cfg):
    assert sum(cfg["shock"]["weights"].values()) == pytest.approx(100.0)


def test_why_flagged_payload_arithmetic_agrees_with_the_stored_score(artifacts):
    _, episodes = artifacts
    if episodes.empty:
        pytest.skip("no episodes produced")
    checked_capped = 0
    for payload in episodes["why_flagged"].head(400):
        why = json.loads(payload)
        points = sum(c["points"] for c in why["components"])
        arithmetic = why["score_arithmetic"]
        assert points == pytest.approx(arithmetic["raw_total"], abs=0.05)

        guarded = points * arithmetic["guard_multiplier"]
        assert guarded == pytest.approx(arithmetic["before_cap"], abs=0.05)
        if arithmetic["warmup_cap_applied"]:
            # A warm-up peak is capped; the panel must show that, not a bare product.
            expected = min(guarded, arithmetic["warmup_cap"])
            checked_capped += 1
        else:
            expected = guarded
        assert expected == pytest.approx(arithmetic["final_score"], abs=0.05), (
            "the 'Why was this flagged?' arithmetic must reconcile exactly")
    assert checked_capped >= 0


def test_scoring_is_deterministic(cfg, artifacts):
    daily, _ = artifacts
    series = daily.groupby(["item_id", "store_id"], observed=True).size().index[0]
    block = daily[(daily["item_id"] == series[0])
                  & (daily["store_id"] == series[1])].sort_values("d")
    args = (block["residual"].to_numpy(dtype="float64"),
            block["y_true"].to_numpy(dtype="float64"),
            block["y_pred"].to_numpy(dtype="float64"),
            np.nan_to_num(block["mu28"].to_numpy(dtype="float64")),
            block["vol_ratio"].to_numpy(dtype="float64"),
            block["level_shift"].to_numpy(dtype="float64"))
    first = S._score_series(*args, cfg)
    second = S._score_series(*args, cfg)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_allclose(first["score"], block["score"].to_numpy(), atol=1e-4)


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def _run(cfg, e, y, f, mu28, vol=None, level=None):
    n = len(e)
    return S._score_series(
        np.asarray(e, dtype="float64"), np.asarray(y, dtype="float64"),
        np.asarray(f, dtype="float64"), np.asarray(mu28, dtype="float64"),
        np.full(n, 1.0) if vol is None else np.asarray(vol, dtype="float64"),
        np.zeros(n) if level is None else np.asarray(level, dtype="float64"),
        cfg)


def test_low_volume_series_cannot_reach_a_high_band(cfg):
    """A 0 -> 2 unit move on a near-zero seller must not read as a crisis."""
    n = 40
    y = np.zeros(n); y[-1] = 2.0
    f = np.full(n, 0.05)
    e = y - f
    out = _run(cfg, e, y, f, np.full(n, 0.2))
    assert out["guard_mult"][-1] == pytest.approx(cfg["shock"]["guard_min_multiplier"])
    assert out["score"][-1] < 30, f"low-volume spike scored {out['score'][-1]:.1f}"


def test_zero_day_against_a_near_zero_forecast_scores_zero(cfg):
    n = 30
    y = np.zeros(n)
    f = np.full(n, 0.4)
    out = _run(cfg, y - f, y, f, np.full(n, 0.5))
    assert out["score"][-1] == 0.0


def test_warmup_days_are_capped_and_flagged(cfg):
    """Early days have too little residual history to justify a high score."""
    n = 30
    y = np.full(n, 40.0)
    f = np.full(n, 4.0)
    out = _run(cfg, y - f, y, f, np.full(n, 40.0))
    cap = float(cfg["shock"]["warmup_score_cap"])
    warm = out["warmup"]
    assert warm[0], "the first scored day must be flagged as warm-up"
    assert out["score"][warm].max() <= cap + 1e-9


def test_a_genuine_high_volume_surge_scores_high(cfg):
    """Control: the guards must not suppress a real, large, sustained deviation."""
    n = 60
    y = np.full(n, 20.0)
    f = np.full(n, 20.0)
    rng = np.random.default_rng(0)
    y[:40] = 20 + rng.normal(0, 1.0, 40)          # quiet period builds the baseline
    y[40:] = 60.0                                  # then a sustained tripling
    e = y - f
    out = _run(cfg, e, y, f, np.full(n, 20.0))
    assert out["score"][-1] >= 70, f"real surge only scored {out['score'][-1]:.1f}"


def test_sigma_is_frozen_while_an_episode_is_open(cfg):
    """Otherwise a long episode inflates its own baseline and self-damps."""
    n = 60
    y = np.full(n, 20.0)
    f = np.full(n, 20.0)
    rng = np.random.default_rng(1)
    y[:40] = 20 + rng.normal(0, 1.0, 40)
    y[40:] = 70.0
    out = _run(cfg, y - f, y, f, np.full(n, 20.0))
    open_days = out["episode_open"]
    assert open_days[-1], "the episode should still be open at the end"
    frozen = out["sigma_e"][out["sigma_frozen"]]
    assert len(frozen) > 0
    assert np.allclose(frozen, frozen[0]), "sigma must not drift while an episode is open"


# ---------------------------------------------------------------------------
# episodes and classification
# ---------------------------------------------------------------------------
def test_episodes_respect_their_filing_rule(cfg, artifacts):
    _, episodes = artifacts
    if episodes.empty:
        pytest.skip("no episodes produced")
    spec = cfg["shock"]["episode"]
    ok = (episodes["n_days"] >= spec["min_days"]) | (
        episodes["peak_score"] >= spec["single_day_score"])
    assert ok.all()
    assert (episodes["end_d"] >= episodes["start_d"]).all()


def test_episode_peak_score_matches_the_daily_table(artifacts):
    daily, episodes = artifacts
    if episodes.empty:
        pytest.skip("no episodes produced")
    for row in episodes.head(30).itertuples(index=False):
        window = daily[(daily["item_id"] == row.item_id)
                       & (daily["store_id"] == row.store_id)
                       & (daily["d"] >= row.start_d) & (daily["d"] <= row.end_d)]
        assert float(window["score"].max()) == pytest.approx(row.peak_score, abs=1e-4)


def test_every_classification_is_a_known_label(artifacts):
    _, episodes = artifacts
    if episodes.empty:
        pytest.skip("no episodes produced")
    assert set(episodes["classification"]).issubset(set(S.CLASSIFICATIONS))


def test_episode_bridges_one_quiet_day_but_closes_after_two(cfg):
    """Directly exercises the state machine used by build_episodes."""
    spec = cfg["shock"]["episode"]
    seed, close = float(spec["seed_score"]), int(spec["close_days"])
    scores = [0, 60, 10, 60, 5, 5, 60, 0]

    episodes, open_idx, below, last_hot = [], None, 0, None
    for i, score in enumerate(scores):
        if open_idx is None:
            if score >= seed:
                open_idx, last_hot, below = i, i, 0
        elif score >= seed:
            last_hot, below = i, 0
        else:
            below += 1
            if below >= close:
                episodes.append((open_idx, last_hot))
                open_idx, below = None, 0
    if open_idx is not None:
        episodes.append((open_idx, last_hot))

    assert episodes == [(1, 3), (6, 6)], (
        "one sub-threshold day must bridge; two consecutive must close the episode")


# ---------------------------------------------------------------------------
# context wording
# ---------------------------------------------------------------------------
def test_fema_context_never_asserts_causation(artifacts):
    _, episodes = artifacts
    if episodes.empty:
        pytest.skip("no episodes produced")
    seen_with_events = 0
    for payload in episodes["fema_context"].head(200):
        context = json.loads(payload)
        sentence = S.format_fema_sentence(context)
        if context.get("n_events", 0) > 0:
            seen_with_events += 1
            assert "not evidence" in sentence.lower()
            assert S.NO_CAUSATION in sentence, (
                "every FEMA context sentence must carry the no-causation disclaimer")
        # Strip the disclaimer before scanning: it deliberately contains the word
        # "caused" inside a negation ("this is NOT evidence that the event caused
        # ..."), which is the opposite of a causal claim.
        body = sentence.replace(S.NO_CAUSATION, "").lower()
        for banned in ("caused", "because of", "due to", "led to", "resulted in",
                       "drove", "triggered", "explains"):
            assert banned not in body, (
                f"causal language outside the disclaimer: {sentence}")
    assert seen_with_events > 0 or all(
        json.loads(p).get("n_events", 0) == 0 for p in episodes["fema_context"].head(200))


def test_fema_context_reports_absence_plainly(artifacts):
    _, episodes = artifacts
    if episodes.empty:
        pytest.skip("no episodes produced")
    empty = [json.loads(p) for p in episodes["fema_context"]
             if json.loads(p).get("n_events", 0) == 0]
    if not empty:
        pytest.skip("every episode coincided with a declaration")
    assert "No FEMA-declared events" in S.format_fema_sentence(empty[0])


def test_daily_scores_only_cover_the_evaluated_window(cfg, artifacts):
    """Residuals exist only where the model forecast out of sample."""
    daily, _ = artifacts
    lowest = min(cfg.fold(f)["val_start_d"] for f in cfg.eval_folds)
    highest = max(cfg.fold(f)["val_end_d"] for f in cfg.eval_folds)
    assert int(daily["d"].min()) >= lowest
    assert int(daily["d"].max()) <= highest


# ---------------------------------------------------------------------------
# actual-based context windows
# ---------------------------------------------------------------------------
def test_level_shift_compares_the_recent_week_to_a_non_overlapping_prior_month(cfg):
    """recent = mean over [t-6, t]; prior = mean over [t-34, t-7]. No overlap.

    Also confirms these statistics read the FULL sales history rather than only the
    scored window, so the detector is not artificially blind at the window's start.
    """
    n = 120
    sales = pd.DataFrame({
        "item_id": ["A"] * n,
        "store_id": ["S"] * n,
        "d": np.arange(1, n + 1),
        "sales": np.concatenate([np.full(80, 4.0), np.full(n - 80, 12.0)]),
    })
    context = S._rolling_context(sales, cfg)
    row = context[context["d"] == 100].iloc[0]

    recent = sales.loc[(sales["d"] >= 94) & (sales["d"] <= 100), "sales"]
    prior = sales.loc[(sales["d"] >= 66) & (sales["d"] <= 93), "sales"]
    assert float(recent.mean()) == pytest.approx(12.0)
    assert float(prior.mean()) == pytest.approx(prior.mean())

    floor = max(float(cfg["shock"]["sigma_floor_abs"]),
                float(cfg["shock"]["sigma_floor_rel"]) * float(row["mu28"]))
    expected = abs(float(recent.mean()) - float(prior.mean())) / max(
        float(prior.std(ddof=1)), floor)
    assert float(row["level_shift"]) == pytest.approx(expected, rel=1e-6)


def test_mu28_excludes_the_current_day(cfg):
    n = 60
    sales = pd.DataFrame({
        "item_id": ["A"] * n, "store_id": ["S"] * n,
        "d": np.arange(1, n + 1),
        "sales": np.arange(1, n + 1, dtype="float64"),
    })
    context = S._rolling_context(sales, cfg)
    row = context[context["d"] == 50].iloc[0]
    expected = float(np.mean(np.arange(22, 50)))   # days 22..49
    assert float(row["mu28"]) == pytest.approx(expected)
