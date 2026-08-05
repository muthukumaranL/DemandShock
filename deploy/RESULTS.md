# DemandShock - measured results

*Generated 2026-08-05T18:50:02+00:00 from the artifacts in `artifacts/`. Every figure below is produced by `scripts/export_results.py` reading files on disk - none is typed by hand.*

**Run mode:** `development` &nbsp;|&nbsp; **trained:** 2026-08-05T18:49:39+00:00 &nbsp;|&nbsp; **config hash:** `e7086b4468a5`

## Data actually processed

| Quantity | Value |
| --- | --- |
| Real series (item x store) | 200 |
| Item-days after release filtering | 329,624 |
| Weekly price records | 48,032 |
| Stores / categories / departments | 1 / 1 / 1 |
| Date coverage | 2011-01-29 to 2016-06-19 (1969 days) |
| Zero-sale share of item-days | 52.5% |
| FEMA disasters in window (CA/TX/WI) | 116 |
| FRED monthly observations | 1,818 |
| Data-quality checks | 63 passed, 0 warnings, 0 failures |

## Validation design

Chronological rolling-origin only. No random splitting anywhere.

| Split | Days | Dates | Role |
| --- | --- | --- | --- |
| F1 | d_1830-d_1857 | 2016-02-01 to 2016-02-28 | backtest fold, ablation, residual pool |
| F2 | d_1858-d_1885 | 2016-02-29 to 2016-03-27 | backtest fold, ablation, residual pool |
| F3 | d_1886-d_1913 | 2016-03-28 to 2016-04-24 | backtest fold, ablation, residual pool |
| HOLDOUT | d_1914-d_1941 | 2016-04-25 to 2016-05-22 | opened once, after selection - headline numbers |
| FORWARD | d_1942-d_1969 | 2016-05-23 to 2016-06-19 | forward forecast beyond the data (no actuals exist) |

## Held-out accuracy (28-day horizon)

| Model | WAPE | RMSSE | MAE | RMSE | Bias | sMAPE |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM (global, Tweedie) - feature set C | 0.8672 | 0.8342 | 1.3448 | 2.3628 | -0.0696 | 124.1 |
| Seasonal naive (lag 28) | 1.0210 | 1.0664 | 1.5832 | 2.8588 | -0.1702 | 102.2 |
| Naive (last observed day) | 1.0428 | 1.0099 | 1.6171 | 2.9119 | -0.0392 | 98.3 |
| Seasonal naive (lag 7) | 1.0438 | 1.0501 | 1.6186 | 2.9023 | -0.0640 | 102.4 |

LightGBM reduces WAPE by **15.1%** against seasonal-naive-28, the benchmark that shares its exact information set. Its RMSSE of 0.8342 is below 1.0, meaning it also beats a one-day naive forecast measured on each series' own training history.

> sMAPE is reported for completeness but is misleading on intermittent demand: on a zero-sale day a naive forecast of exactly zero scores perfectly while any positive forecast is penalised the full 200%. WAPE and RMSSE are the trustworthy comparisons here.

### By horizon

| Horizon | WAPE | RMSSE | Bias |
| --- | --- | --- | --- |
| 7 days | 0.8367 | 0.7459 | -0.0559 |
| 14 days | 0.8582 | 0.7870 | -0.0555 |
| 28 days | 0.8672 | 0.8342 | -0.0696 |

**Prediction intervals.** Empirical P10-P90 bands covered 75.6% of held-out actuals against an 80% design target (12.2% fell below P10, 12.2% above P90, n=5,600).

## Ablation: do FEMA and FRED actually help?

Mean across folds F1, F2, F3 at horizon 28, identical seed, parameters and rows in every arm.

| Arm | Feature set | Features | WAPE | sd across folds | RMSSE | Bias |
| --- | --- | --- | --- | --- | --- | --- |
| A | Demand history only | 19 | 0.8909 | 0.0512 | 0.7133 | 0.0276 |
| B | + Calendar & Price | 36 | 0.8839 | 0.0398 | 0.7069 | 0.0454 |
| C | + FEMA disaster context | 43 | 0.8796 | 0.0420 | 0.7043 | 0.0453 |
| D | + FRED economic context | 47 | 0.8906 | 0.0385 | 0.7091 | 0.0435 |

- **Calendar and price: improved accuracy.** WAPE fell 0.0070 (0.78% relative), better on 2 of 3 folds. This is smaller than the 0.0512 fold-to-fold spread, so it is a marginal gain, not a decisive one.
- **FEMA disaster context: improved accuracy.** WAPE fell 0.0042 (0.48% relative), better on 3 of 3 folds. This is smaller than the 0.0398 fold-to-fold spread, so it is a marginal gain, not a decisive one.
- **FRED economic context: did NOT improve accuracy.** WAPE rose 0.0110 (1.25% relative), better on only 0 of 3 folds.

For perspective, FEMA features account for 0.06% of total model gain. Read the WAPE deltas above against those shares before concluding that external data improves point forecasts.

FRED features are not part of the selected feature set, so they carry no attribution here by construction - the ablation table above is the evidence on whether they help.

**Share of model gain by feature family**

| Family | Share of gain |
| --- | --- |
| Demand history | 80.05% |
| Product / Store identity | 15.47% |
| Calendar & Events | 3.10% |
| Price | 1.32% |
| FEMA context | 0.06% |

Top features by gain: `roll_mean_28`, `roll_mean_56`, `item_id`, `roll_mean_7`, `wday`, `days_since_last_sale`.

**Selection.** smallest feature set within 0.3% relative of the best mean WAPE at h=28. Selected feature set **C** (+ FEMA disaster context).

**Objective check** (development mode, fold F1, arm B, then frozen):

| Objective | WAPE | RMSSE | Bias |
| --- | --- | --- | --- |
| tweedie | 0.9426 | 0.7525 | 0.0451 |
| poisson | 0.9484 | 0.7564 | 0.0421 |
| regression | 0.9498 | 0.7602 | 0.0398 |

## Demand shock detection

Scored window: **2016-02-01 to 2016-05-22** (112 days), covering 170 eligible series and 19,040 scored series-days. Residuals come from feature set B, which contains no FEMA or FRED features by design.

**421 episodes** were filed.

| Classification | Episodes | Median duration | Mean peak score |
| --- | --- | --- | --- |
| Volatility Shock | 169 | 4 days | 51.3 |
| Demand Surge | 133 | 3 days | 55.5 |
| Demand Collapse | 94 | 5 days | 55.2 |
| Persistent Over-forecast | 19 | 8 days | 48.2 |
| Regime Shift | 5 | 33 days | 76.8 |
| Possible Regime Shift (window truncated) | 1 | 46 days | 79.8 |

| Severity | Episodes |
| --- | --- |
| Critical | 1 |
| Severe | 39 |
| Elevated | 201 |
| Watch | 180 |

**0.0%** of episodes coincided with a FEMA declaration active in the same state. This is a coincidence rate measured over the scored window - it is not evidence of causation, and the platform never presents it as such.

## Business impact (measured, not assumed)

| Quantity | Value |
| --- | --- |
| Series covered | 200 (200 eligible for planning arithmetic) |
| Actual units sold in the holdout window | 8,684 |
| Revenue represented at real M5 prices | $22,777 |
| Under-forecast exposure | $11,610 |
| Over-forecast exposure | $9,694 |
| Total estimated revenue exposure | $21,304 |

Estimated revenue exposure dollarises forecast error using real M5 sell prices. It is **not** measured lost revenue: M5 records units sold, so demand that was never satisfied is unobservable in the source data. Safety stock, reorder points and days of cover are shown in the application only after a user supplies lead time and service level, because M5 contains no inventory records at all.

## Reproducing these numbers

```bash
python scripts/run_pipeline.py --mode development
python scripts/export_results.py
```

Library versions: python 3.14.4, pandas 3.0.2, numpy 2.4.4, lightgbm 4.6.0, pyarrow 23.0.1, scipy 1.17.1, scikit-learn 1.8.0.
