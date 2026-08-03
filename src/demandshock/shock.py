"""Demand-shock detection - the module DemandShock is named for.

WHAT A SHOCK IS HERE
--------------------
Not "high demand". A shock is a day where ACTUAL demand departed from what the
model could reasonably have expected at the forecast origin, in a way that is
large relative to that series' own recent forecast error, and/or persistent,
and/or accompanied by a change in volatility or demand level.

Residuals only exist where the model forecast out-of-sample, so shock detection
runs on the historical evaluated window (folds F1-F3 plus the holdout). This is a
retrospective diagnostic, and the app says so on the page.

WHY THE RESIDUALS COME FROM ARM B
---------------------------------
Arm B has demand, calendar and price features but NO FEMA or FRED features. If
crisis features were in the model, crisis-driven deviations would be partly
absorbed into the prediction and would vanish from the residual - the detector
would be blinded to exactly what it exists to find. Keeping crisis signals OUT of
the model keeps them IN the residual.

SCORE CONSTRUCTION
------------------
Five signals, each squashed to [0,1] by a piecewise-linear cap, then weighted to
sum to 100. Piecewise-linear (not a sigmoid) is deliberate: every point of the
score is attributable to a component and the arithmetic is checkable by hand in
the "Why was this flagged?" panel.

The score is a project-defined metric. It is NOT an industry standard.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import Config

COMPONENTS = ["std_residual", "pct_deviation", "persistence",
              "volatility_ratio", "level_shift"]

CLASSIFICATIONS = [
    "Regime Shift",
    "Possible Regime Shift (window truncated)",
    "Persistent Under-forecast",
    "Persistent Over-forecast",
    "Demand Surge",
    "Demand Collapse",
    "Volatility Shock",
]

BAND_ORDER = ["Normal", "Watch", "Elevated", "Severe", "Critical"]


def band_for(score: float, cfg: Config) -> str:
    for name in BAND_ORDER:
        lo, hi = cfg["shock"]["bands"][name]
        if lo <= score < hi:
            return name
    return BAND_ORDER[-1]


def _band_array(scores: np.ndarray, cfg: Config) -> np.ndarray:
    edges, labels = [], []
    for name in BAND_ORDER:
        lo, _ = cfg["shock"]["bands"][name]
        edges.append(lo)
        labels.append(name)
    idx = np.searchsorted(np.array(edges[1:]), scores, side="right")
    return np.array(labels, dtype=object)[idx]


# ---------------------------------------------------------------------------
# per-series scoring
# ---------------------------------------------------------------------------
def _score_series(e: np.ndarray, y: np.ndarray, f: np.ndarray, mu28: np.ndarray,
                  vol_ratio: np.ndarray, level_shift: np.ndarray,
                  cfg: Config) -> dict[str, np.ndarray]:
    """Sequential pass over one series' scored days.

    Sequential (not vectorised) because the z-score denominator is FROZEN at its
    pre-episode value while an episode is open. Without that freeze a long episode
    inflates its own rolling residual standard deviation and silently damps its
    later days - the baseline would be contaminated by the very period being scored.
    """
    spec = cfg["shock"]
    weights = spec["weights"]
    squash = spec["squash"]
    n = len(e)

    sigma_floor_abs = float(spec["sigma_floor_abs"])
    sigma_floor_rel = float(spec["sigma_floor_rel"])
    pct_floor = float(spec["pct_deviation_floor"])
    guard_ref = float(spec["guard_volume_ref"])
    guard_min = float(spec["guard_min_multiplier"])
    warmup_days = int(spec["warmup_days"])
    warmup_cap = float(spec["warmup_score_cap"])
    seed_score = float(spec["episode"]["seed_score"])
    close_days = int(spec["episode"]["close_days"])

    out = {
        "sigma_e": np.full(n, np.nan), "z": np.zeros(n), "pct_dev": np.zeros(n),
        "persistence_run": np.zeros(n, dtype="int16"),
        "sq_z": np.zeros(n), "sq_pct": np.zeros(n), "sq_pers": np.zeros(n),
        "sq_vol": np.zeros(n), "sq_level": np.zeros(n),
        "score_raw": np.zeros(n), "guard_mult": np.ones(n),
        "score": np.zeros(n), "warmup": np.zeros(n, dtype=bool),
        "episode_open": np.zeros(n, dtype=bool), "sigma_frozen": np.zeros(n, dtype=bool),
    }

    quiet: list[float] = []          # residuals from days outside an open episode
    episode_open = False
    frozen_sigma = np.nan
    below_run = 0
    run_length = 0
    run_sign = 0

    for i in range(n):
        # --- residual dispersion (the z denominator) ----------------------
        if episode_open and np.isfinite(frozen_sigma):
            sigma = frozen_sigma
            out["sigma_frozen"][i] = True
        else:
            recent = quiet[-28:]
            sigma = float(np.std(recent, ddof=1)) if len(recent) >= 2 else np.nan
        floor = max(sigma_floor_abs, sigma_floor_rel * float(mu28[i]))
        sigma = floor if not np.isfinite(sigma) else max(sigma, floor)
        out["sigma_e"][i] = sigma

        is_warmup = len(quiet) < warmup_days and not episode_open
        out["warmup"][i] = is_warmup

        # --- components ---------------------------------------------------
        z = float(e[i]) / sigma if sigma > 0 else 0.0
        pct = abs(float(e[i])) / max(float(mu28[i]), pct_floor)

        sign = 1 if e[i] > 0 else (-1 if e[i] < 0 else 0)
        if sign != 0 and abs(z) >= 1.0 and sign == run_sign:
            run_length += 1
        elif sign != 0 and abs(z) >= 1.0:
            run_length, run_sign = 1, sign
        else:
            run_length, run_sign = 0, 0

        sq_z = float(np.clip(abs(z) / float(squash["z_saturate"]), 0, 1))
        sq_pct = float(np.clip(pct / float(squash["pct_saturate"]), 0, 1))
        sq_pers = float(np.clip(run_length / float(squash["persistence_saturate"]), 0, 1))
        sq_vol = float(np.clip(
            (vol_ratio[i] - 1.0) / (float(squash["volatility_saturate"]) - 1.0), 0, 1)
            if np.isfinite(vol_ratio[i]) else 0.0)
        sq_level = float(np.clip(level_shift[i] / float(squash["level_saturate"]), 0, 1)
                         if np.isfinite(level_shift[i]) else 0.0)

        raw = (weights["std_residual"] * sq_z
               + weights["pct_deviation"] * sq_pct
               + weights["persistence"] * sq_pers
               + weights["volatility_ratio"] * sq_vol
               + weights["level_shift"] * sq_level)

        # --- guards -------------------------------------------------------
        base_vol = max(float(mu28[i]), float(f[i]))
        guard = float(np.clip(base_vol / guard_ref, guard_min, 1.0))
        score = raw * guard
        if y[i] == 0 and f[i] < 1:
            score = 0.0        # a zero day against a near-zero forecast is not news
        if is_warmup:
            score = min(score, warmup_cap)

        out["z"][i] = z
        out["pct_dev"][i] = pct
        out["persistence_run"][i] = run_length
        out["sq_z"][i], out["sq_pct"][i] = sq_z, sq_pct
        out["sq_pers"][i], out["sq_vol"][i], out["sq_level"][i] = sq_pers, sq_vol, sq_level
        out["score_raw"][i] = raw
        out["guard_mult"][i] = guard
        out["score"][i] = score

        # --- episode state machine ---------------------------------------
        if not episode_open:
            if score >= seed_score:
                episode_open = True
                frozen_sigma = sigma       # already computed from pre-episode days
                below_run = 0
            else:
                quiet.append(float(e[i]))
        else:
            if score < seed_score:
                below_run += 1
                if below_run >= close_days:
                    episode_open = False
                    frozen_sigma = np.nan
                    below_run = 0
                    quiet.append(float(e[i]))
            else:
                below_run = 0
        out["episode_open"][i] = episode_open
    return out


def _rolling_context(sales: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Actual-based statistics, computed over FULL history, not just the scored window.

    mu28 and the level-shift signal use observed sales, which exist long before the
    scored window begins - so there is no artificial blind spot at its start.
    """
    sales = sales.sort_values(["item_id", "store_id", "d"]).copy()
    grp = sales.groupby(["item_id", "store_id"], observed=True)["sales"]
    shifted = grp.shift(1)
    by = [sales["item_id"], sales["store_id"]]

    sales["mu28"] = (shifted.groupby(by, observed=True)
                     .rolling(28, min_periods=5).mean()
                     .reset_index(level=[0, 1], drop=True).to_numpy())
    recent = (grp.rolling(7, min_periods=4).mean()
              .reset_index(level=[0, 1], drop=True).to_numpy())
    prior_mean = (shifted.groupby(by, observed=True)
                  .rolling(28, min_periods=8).mean()
                  .reset_index(level=[0, 1], drop=True).to_numpy())
    prior_std = (shifted.groupby(by, observed=True)
                 .rolling(28, min_periods=8).std()
                 .reset_index(level=[0, 1], drop=True).to_numpy())
    # prior window is shifted a further 6 days so it does not overlap the recent week
    prior_mean = pd.Series(prior_mean).groupby(
        [s.reset_index(drop=True) for s in by]).shift(6).to_numpy()
    prior_std = pd.Series(prior_std).groupby(
        [s.reset_index(drop=True) for s in by]).shift(6).to_numpy()

    floor = np.maximum(float(cfg["shock"]["sigma_floor_abs"]),
                       float(cfg["shock"]["sigma_floor_rel"]) * np.nan_to_num(sales["mu28"]))
    with np.errstate(invalid="ignore", divide="ignore"):
        sales["level_shift"] = np.abs(recent - prior_mean) / np.maximum(prior_std, floor)
    sales["mu28"] = sales["mu28"].fillna(0.0)
    return sales[["item_id", "store_id", "d", "mu28", "level_shift"]]


def score_daily(cfg: Config, residuals: pd.DataFrame, sales: pd.DataFrame,
                logger=None) -> pd.DataFrame:
    """Daily shock scores for every eligible series in the evaluated window."""
    spec = cfg["shock"]
    elig = spec["eligibility"]

    residuals = residuals.dropna(subset=["y_true", "y_pred"]).copy()
    residuals["item_id"] = residuals["item_id"].astype(str)
    residuals["store_id"] = residuals["store_id"].astype(str)
    residuals = residuals.sort_values(["item_id", "store_id", "d"]).reset_index(drop=True)
    residuals["residual"] = residuals["y_true"] - residuals["y_pred"]

    context = _rolling_context(sales, cfg)
    residuals = residuals.merge(context, on=["item_id", "store_id", "d"],
                                how="left", validate="1:1")

    # volatility ratio uses residuals inside the scored window (it is meant to react)
    by = [residuals["item_id"], residuals["store_id"]]
    r7 = (residuals.groupby(["item_id", "store_id"], observed=True)["residual"]
          .rolling(7, min_periods=4).std().reset_index(level=[0, 1], drop=True))
    r28 = (residuals.groupby(["item_id", "store_id"], observed=True)["residual"]
           .rolling(28, min_periods=8).std().reset_index(level=[0, 1], drop=True))
    floor = np.maximum(float(spec["sigma_floor_abs"]),
                       float(spec["sigma_floor_rel"]) * residuals["mu28"].to_numpy())
    with np.errstate(invalid="ignore", divide="ignore"):
        residuals["vol_ratio"] = r7.to_numpy() / np.maximum(r28.to_numpy(), floor)

    # --- eligibility ------------------------------------------------------
    first_scored_d = int(residuals["d"].min())
    trailing = sales[(sales["d"] < first_scored_d)
                     & (sales["d"] >= first_scored_d - 56)]
    nonzero = (trailing.assign(nz=(trailing["sales"] > 0).astype(int))
               .groupby(["item_id", "store_id"], observed=True)["nz"].sum()
               .rename("nonzero_56").reset_index())
    counts = (residuals.groupby(["item_id", "store_id"], observed=True)
              .size().rename("residual_days").reset_index())
    gate = counts.merge(nonzero, on=["item_id", "store_id"], how="left")
    gate["nonzero_56"] = gate["nonzero_56"].fillna(0)
    gate["eligible"] = (
        (gate["residual_days"] >= int(elig["min_residual_days"]))
        & (gate["nonzero_56"] >= int(elig["min_nonzero_days_trailing_56"])))
    residuals = residuals.merge(gate, on=["item_id", "store_id"], how="left")

    eligible = residuals[residuals["eligible"]].copy()
    excluded = int((~gate["eligible"]).sum())
    if logger:
        logger.info("    %s of %s series eligible for scoring (%s excluded: "
                    "too few non-zero days or too little residual history)",
                    f"{int(gate['eligible'].sum()):,}", f"{len(gate):,}", f"{excluded:,}")
    if eligible.empty:
        return eligible.assign(score=[], band=[])

    parts = []
    for (item_id, store_id), block in eligible.groupby(
            ["item_id", "store_id"], observed=True, sort=False):
        block = block.sort_values("d")
        result = _score_series(
            block["residual"].to_numpy(dtype="float64"),
            block["y_true"].to_numpy(dtype="float64"),
            block["y_pred"].to_numpy(dtype="float64"),
            np.nan_to_num(block["mu28"].to_numpy(dtype="float64")),
            block["vol_ratio"].to_numpy(dtype="float64"),
            block["level_shift"].to_numpy(dtype="float64"),
            cfg)
        scored = block.copy()
        for key, values in result.items():
            scored[key] = values
        parts.append(scored)

    daily = pd.concat(parts, ignore_index=True)
    daily["band"] = _band_array(daily["score"].to_numpy(), cfg)
    daily["score"] = daily["score"].astype("float32")
    return daily


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------
def build_episodes(cfg: Config, daily: pd.DataFrame, calendar: pd.DataFrame,
                   meta: pd.DataFrame, sales: pd.DataFrame | None = None) -> pd.DataFrame:
    """Group flagged days into episodes and classify each one."""
    spec = cfg["shock"]["episode"]
    rules = cfg["shock"]["classification"]
    seed = float(spec["seed_score"])
    close_days = int(spec["close_days"])
    min_days = int(spec["min_days"])
    single_day = float(spec["single_day_score"])

    day_to_date = dict(zip(calendar["d"], calendar["date"]))
    hierarchy = meta.set_index(["item_id", "store_id"])[
        ["dept_id", "cat_id", "state_id"]]

    # Baseline demand must come from the FULL sales history, not just the scored
    # window: an episode starting on day 3 of the window would otherwise be judged
    # against a two-day baseline.
    history: dict[tuple[str, str], pd.Series] = {}
    if sales is not None:
        for key, block in sales.groupby(["item_id", "store_id"], observed=True, sort=False):
            history[(str(key[0]), str(key[1]))] = pd.Series(
                block["sales"].to_numpy(dtype="float64"),
                index=block["d"].to_numpy())

    episodes: list[dict[str, Any]] = []
    for (item_id, store_id), block in daily.groupby(
            ["item_id", "store_id"], observed=True, sort=False):
        block = block.sort_values("d").reset_index(drop=True)
        actuals = history.get((str(item_id), str(store_id)))
        scores = block["score"].to_numpy()

        open_idx = None
        below = 0
        last_hot = None
        for i, score in enumerate(scores):
            if open_idx is None:
                if score >= seed:
                    open_idx, last_hot, below = i, i, 0
            else:
                if score >= seed:
                    last_hot, below = i, 0
                else:
                    below += 1
                    if below >= close_days:
                        episodes.append(
                            _episode_record(cfg, block, open_idx, last_hot,
                                            item_id, store_id, hierarchy,
                                            day_to_date, rules, actuals))
                        open_idx, below = None, 0
        if open_idx is not None:
            episodes.append(_episode_record(cfg, block, open_idx, last_hot,
                                            item_id, store_id, hierarchy,
                                            day_to_date, rules, actuals))

    if not episodes:
        return pd.DataFrame(columns=[
            "episode_id", "item_id", "store_id", "dept_id", "cat_id", "state_id",
            "start_date", "end_date", "start_d", "end_d", "n_days", "peak_score",
            "mean_score", "band", "classification", "dominant_component",
            "sign_share_pos", "peak_z", "max_run", "total_residual",
            "why_flagged"])

    frame = pd.DataFrame(episodes)
    keep = (frame["n_days"] >= min_days) | (frame["peak_score"] >= single_day)
    return frame[keep].reset_index(drop=True)


def _episode_record(cfg: Config, block: pd.DataFrame, start_i: int, end_i: int,
                    item_id: str, store_id: str, hierarchy: pd.DataFrame,
                    day_to_date: dict, rules: dict,
                    actuals: pd.Series | None = None) -> dict[str, Any]:
    window = block.iloc[start_i:end_i + 1]
    peak_i = int(window["score"].idxmax())
    peak = block.loc[peak_i]
    weights = cfg["shock"]["weights"]

    points = {
        "std_residual": weights["std_residual"] * float(peak["sq_z"]),
        "pct_deviation": weights["pct_deviation"] * float(peak["sq_pct"]),
        "persistence": weights["persistence"] * float(peak["sq_pers"]),
        "volatility_ratio": weights["volatility_ratio"] * float(peak["sq_vol"]),
        "level_shift": weights["level_shift"] * float(peak["sq_level"]),
    }
    dominant = max(points, key=points.get)
    residuals = window["residual"].to_numpy()
    sign_share_pos = float((residuals > 0).mean())
    max_run = int(window["persistence_run"].max())
    peak_z = float(peak["z"])

    # --- classification (ordered decision tree, first match wins) ---------
    # A regime shift is a LASTING change in the demand level: a long episode whose
    # late-window demand sits far from the pre-episode baseline, confirmed by the
    # days that follow where those days exist. Duration + level change decide it -
    # not which component happened to dominate the single peak day, which the
    # residual term always wins.
    start_d, end_d = int(window["d"].min()), int(window["d"].max())
    lookahead = int(rules["regime_lookahead_days"])
    min_regime_days = int(rules["regime_min_days"])

    baseline = late = after_values = np.array([])
    if actuals is not None and len(actuals):
        baseline = actuals[(actuals.index >= start_d - 28) & (actuals.index < start_d)].to_numpy()
        late = window["y_true"].to_numpy()[-min(lookahead, len(window)):]
        after_values = actuals[(actuals.index > end_d)
                               & (actuals.index <= end_d + lookahead)].to_numpy()

    denom = max(float(np.std(baseline, ddof=1)) if baseline.size > 1 else 0.0,
                float(cfg["shock"]["sigma_floor_abs"]))
    baseline_mean = float(baseline.mean()) if baseline.size else np.nan
    level_change = (abs(float(late.mean()) - baseline_mean) / denom
                    if baseline.size and late.size else np.nan)
    after_change = (abs(float(after_values.mean()) - baseline_mean) / denom
                    if baseline.size and after_values.size else np.nan)
    truncated = after_values.size < lookahead
    threshold = float(rules["regime_persist_ratio"])
    rel_threshold = float(rules["regime_relative_change"])

    def _relative(values: np.ndarray) -> float:
        """Sustained change as a share of the baseline level.

        Needed because scaling by the baseline's own standard deviation understates
        a drop to zero on low-volume, noisy series: an item selling ~2 units a day
        with sigma 1.5 that then records 80 straight zeros scores only ~1.3 sigma,
        yet it has plainly stopped selling.
        """
        if not values.size or not np.isfinite(baseline_mean) or baseline_mean <= 0:
            return np.nan
        return abs(float(values.mean()) - baseline_mean) / baseline_mean

    rel_change = _relative(late)
    rel_after = _relative(after_values)

    def _shifted(sigma_ratio: float, relative: float) -> bool:
        return ((np.isfinite(sigma_ratio) and sigma_ratio >= threshold)
                or (np.isfinite(relative) and relative >= rel_threshold))

    is_regime = (
        len(window) >= min_regime_days
        and _shifted(level_change, rel_change)
        and (truncated or _shifted(after_change, rel_after))
    )

    trace: list[str] = []
    if is_regime:
        classification = ("Possible Regime Shift (window truncated)" if truncated
                          else "Regime Shift")
        confirm = ("no post-episode window to confirm persistence" if truncated
                   else "confirmed by demand after the episode")
        trace.append(
            f"regime_shift: yes (duration={len(window)} >= {min_regime_days} days, "
            f"level change {level_change:.2f} sigma / {rel_change:.0%} of baseline, "
            f"{confirm})")
    else:
        change_txt = f"{level_change:.2f} sigma" if np.isfinite(level_change) else "n/a"
        rel_txt = f" / {rel_change:.0%} of baseline" if np.isfinite(rel_change) else ""
        reason = ("duration below threshold" if len(window) < min_regime_days
                  else ("level change did not persist after the episode"
                        if _shifted(level_change, rel_change)
                        else "level change below threshold"))
        trace.append(f"regime_shift: no ({reason}; duration={len(window)}, "
                     f"level change {change_txt}{rel_txt})")
        if max_run >= int(rules["persistence_run"]) and abs(peak_z) < float(rules["persistent_max_z"]):
            classification = ("Persistent Under-forecast" if sign_share_pos >= 0.5
                              else "Persistent Over-forecast")
            trace.append(f"persistence: yes (run={max_run} >= {rules['persistence_run']}, "
                         f"|peak z|={abs(peak_z):.2f} < {rules['persistent_max_z']})")
        else:
            trace.append(f"persistence: no (run={max_run}, |peak z|={abs(peak_z):.2f})")
            share = max(sign_share_pos, 1 - sign_share_pos)
            if share >= float(rules["surge_sign_share"]) and abs(peak_z) >= float(rules["surge_min_z"]):
                classification = "Demand Surge" if sign_share_pos >= 0.5 else "Demand Collapse"
                trace.append(f"surge/collapse: yes (sign_share={share:.2f} >= "
                             f"{rules['surge_sign_share']}, |peak z|={abs(peak_z):.2f} >= "
                             f"{rules['surge_min_z']})")
            else:
                classification = "Volatility Shock"
                trace.append(f"surge/collapse: no (sign_share={share:.2f}, "
                             f"|peak z|={abs(peak_z):.2f}) -> volatility shock")

    try:
        hier = hierarchy.loc[(item_id, store_id)]
        dept, cat, state = hier["dept_id"], hier["cat_id"], hier["state_id"]
    except KeyError:
        dept = cat = state = None

    start_date = day_to_date[int(window["d"].min())]
    end_date = day_to_date[int(window["d"].max())]
    why = {
        "episode_id": f"{item_id}__{store_id}__{start_date.date()}",
        "peak_date": str(day_to_date[int(peak['d'])].date()),
        "peak_score": round(float(peak["score"]), 2),
        "band": str(peak["band"]),
        "classification": classification,
        "rule_trace": trace,
        "components": [
            {"name": name,
             "raw": round(float(peak[raw_col]), 3),
             "squashed": round(float(peak[sq_col]), 3),
             "points": round(points[name], 2),
             "weight": weights[name]}
            for name, raw_col, sq_col in [
                ("std_residual", "z", "sq_z"),
                ("pct_deviation", "pct_dev", "sq_pct"),
                ("persistence", "persistence_run", "sq_pers"),
                ("volatility_ratio", "vol_ratio", "sq_vol"),
                ("level_shift", "level_shift", "sq_level")]
        ],
        "score_arithmetic": {
            "raw_total": round(float(peak["score_raw"]), 2),
            "guard_multiplier": round(float(peak["guard_mult"]), 3),
            "final_score": round(float(peak["score"]), 2),
            "note": ("final = raw x guard multiplier; the guard damps low-volume "
                     "series so a 0->2 unit move cannot read as a crisis"),
        },
        "level_change_vs_baseline": (round(float(level_change), 2)
                                     if np.isfinite(level_change) else None),
        "sigma_frozen_during_episode": bool(peak["sigma_frozen"]),
        "warmup": bool(peak["warmup"]),
        "residual_summary": {
            "mean_residual": round(float(residuals.mean()), 2),
            "sign_share_positive": round(sign_share_pos, 2),
            "days": int(len(window)),
        },
    }

    return {
        "episode_id": why["episode_id"],
        "item_id": item_id, "store_id": store_id,
        "dept_id": dept, "cat_id": cat, "state_id": state,
        "start_d": int(window["d"].min()), "end_d": int(window["d"].max()),
        "start_date": start_date, "end_date": end_date,
        "n_days": int(len(window)),
        "peak_score": float(peak["score"]), "mean_score": float(window["score"].mean()),
        "band": str(peak["band"]), "classification": classification,
        "dominant_component": dominant, "sign_share_pos": sign_share_pos,
        "peak_z": peak_z, "max_run": max_run,
        "total_residual": float(residuals.sum()),
        "why_flagged": json.dumps(why),
    }


# ---------------------------------------------------------------------------
# real-world context (association only, never causation)
# ---------------------------------------------------------------------------
NO_CAUSATION = (
    "State-level temporal coincidence only - this is not evidence that the event "
    "caused the demand deviation. FEMA declarations are county-scoped and M5 does "
    "not disclose store locations within a state."
)


def attach_context(cfg: Config, episodes: pd.DataFrame, fema_context: pd.DataFrame,
                   fred_daily: pd.DataFrame) -> pd.DataFrame:
    """Attach coinciding FEMA declarations and economic conditions to each episode."""
    if episodes.empty:
        episodes["fema_context"] = []
        episodes["econ_context"] = []
        episodes["fema_overlap"] = []
        return episodes

    lead = pd.Timedelta(days=int(cfg["fema"]["context_lead_days"]))
    fema_context = fema_context.copy()
    fema_context["state_id"] = fema_context["state_id"].astype(str)
    fred_daily = fred_daily.copy()
    fred_daily["state_id"] = fred_daily["state_id"].astype(str)

    fema_payloads, econ_payloads, overlaps = [], [], []
    for row in episodes.itertuples(index=False):
        state = str(row.state_id)
        window_start = row.start_date - lead
        matches = fema_context[
            (fema_context["state_id"] == state)
            & (fema_context["incident_begin"] <= row.end_date)
            & (fema_context["incident_end"] >= window_start)
        ]
        events = []
        for m in matches.itertuples(index=False):
            overlap_days = int(
                (min(m.incident_end, row.end_date)
                 - max(m.incident_begin, row.start_date)).days) + 1
            events.append({
                "disasterNumber": int(m.disasterNumber),
                "declarationType": str(m.declarationType),
                "incidentType": str(m.incidentType),
                "title": str(m.declarationTitle),
                "incident_begin": str(pd.Timestamp(m.incident_begin).date()),
                "incident_end": str(pd.Timestamp(m.incident_end).date()),
                "end_estimated": bool(m.end_imputed),
                "counties_designated": int(m.n_counties),
                "overlap_days": max(overlap_days, 0),
            })
        fema_payloads.append(json.dumps({
            "state": state,
            "n_events": len(events),
            "events": events,
            "disclaimer": NO_CAUSATION,
        }))
        overlaps.append(len(events) > 0)

        econ_rows = fred_daily[(fred_daily["state_id"] == state)
                               & (fred_daily["date"] <= row.start_date)]
        if econ_rows.empty:
            econ_payloads.append(json.dumps({"available": False}))
            continue
        latest = econ_rows.iloc[-1]
        prior = fred_daily[(fred_daily["state_id"] == state)
                           & (fred_daily["date"] <= row.start_date - pd.Timedelta(days=90))]
        prior_value = float(prior.iloc[-1]["ur_level"]) if not prior.empty else float("nan")
        econ_payloads.append(json.dumps({
            "available": True,
            "state": state,
            "unemployment_rate": float(latest["ur_level"]),
            "source_month": str(pd.Timestamp(latest["source_month"]).date()),
            "change_vs_3m_ago": (None if not np.isfinite(prior_value)
                                 else round(float(latest["ur_level"]) - prior_value, 2)),
            "disclaimer": ("Macroeconomic context only. Monthly data cannot explain "
                           "a daily deviation and no causal link is implied."),
        }))

    episodes = episodes.copy()
    episodes["fema_context"] = fema_payloads
    episodes["econ_context"] = econ_payloads
    episodes["fema_overlap"] = overlaps
    return episodes


def format_fema_sentence(payload: dict) -> str:
    """Exact wording used in the UI. Association only - never causal."""
    if not payload or payload.get("n_events", 0) == 0:
        return (f"No FEMA-declared events were active in "
                f"{payload.get('state', 'this state')} during this window.")
    events = payload["events"]
    head = events[0]
    extra = f" (and {len(events) - 1} other declaration(s))" if len(events) > 1 else ""
    estimated = " (end date estimated)" if head.get("end_estimated") else ""
    return (
        f"{len(events)} FEMA-declared event(s) were active in {payload['state']} "
        f"during this episode window: \"{head['title']}\" ({head['incidentType']}, "
        f"{head['declarationType']}, {head['incident_begin']} to "
        f"{head['incident_end']}{estimated}){extra}. {NO_CAUSATION}"
    )
