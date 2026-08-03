# DemandShock - measured results

*Generated 2026-08-03T10:41:56+00:00 from the artifacts in `artifacts/`. Every figure below is produced by `scripts/export_results.py` reading files on disk - none is typed by hand.*

**Run mode:** `full` &nbsp;|&nbsp; **trained:** 2026-08-03T10:09:58+00:00 &nbsp;|&nbsp; **config hash:** `3cb8d141be90`

## Data actually processed

| Quantity | Value |
| --- | --- |
| Real series (item x store) | 30,490 |
| Item-days after release filtering | 46,881,677 |
| Weekly price records | 6,841,121 |
| Stores / categories / departments | 10 / 3 / 7 |
| Date coverage | 2011-01-29 to 2016-06-19 (1969 days) |
| Zero-sale share of item-days | 59.6% |
| FEMA disasters in window (CA/TX/WI) | 116 |
| FRED monthly observations | 1,818 |
| Data-quality checks | 59 passed, 0 warnings, 0 failures |

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
| LightGBM (global, Tweedie) - feature set B | 0.7305 | 0.7606 | 1.0540 | 2.1941 | -0.0762 | 140.3 |
| Seasonal naive (lag 7) | 0.8622 | 0.9976 | 1.2440 | 2.6769 | -0.0736 | 83.1 |
| Seasonal naive (lag 28) | 0.8900 | 1.0338 | 1.2840 | 2.8292 | -0.0391 | 84.0 |
| Naive (last observed day) | 0.9516 | 1.0017 | 1.3730 | 2.8936 | 0.1319 | 85.7 |

LightGBM reduces WAPE by **17.9%** against seasonal-naive-28, the benchmark that shares its exact information set. Its RMSSE of 0.7606 is below 1.0, meaning it also beats a one-day naive forecast measured on each series' own training history.

> sMAPE is reported for completeness but is misleading on intermittent demand: on a zero-sale day a naive forecast of exactly zero scores perfectly while any positive forecast is penalised the full 200%. WAPE and RMSSE are the trustworthy comparisons here.

### By horizon

| Horizon | WAPE | RMSSE | Bias |
| --- | --- | --- | --- |
| 7 days | 0.7436 | 0.6848 | -0.0532 |
| 14 days | 0.7397 | 0.7274 | -0.0527 |
| 28 days | 0.7305 | 0.7606 | -0.0762 |

**Prediction intervals.** Empirical P10-P90 bands covered 79.6% of held-out actuals against an 80% design target (9.7% fell below P10, 10.7% above P90, n=853,624).

## Ablation: do FEMA and FRED actually help?

Mean across folds F1, F2, F3 at horizon 28, identical seed, parameters and rows in every arm.

| Arm | Feature set | Features | WAPE | sd across folds | RMSSE | Bias |
| --- | --- | --- | --- | --- | --- | --- |
| A | Demand history only | 19 | 0.7752 | 0.0078 | 0.7480 | -0.0581 |
| B | + Calendar & Price | 36 | 0.7635 | 0.0087 | 0.7393 | -0.0437 |
| C | + FEMA disaster context | 43 | 0.7632 | 0.0100 | 0.7390 | -0.0423 |
| D | + FRED economic context | 47 | 0.7633 | 0.0100 | 0.7386 | -0.0348 |

- **Calendar and price: improved accuracy.** WAPE fell 0.0117 (1.51% relative), better on 3 of 3 folds.
- **FEMA disaster context: improved accuracy.** WAPE fell 0.0004 (0.05% relative), better on 1 of 3 folds. This is smaller than the 0.0087 fold-to-fold spread, so it is a marginal gain, not a decisive one.
- **FRED economic context: did NOT improve accuracy.** WAPE rose 0.0001 (0.01% relative), better on only 1 of 3 folds.

For perspective, FEMA features account for 0.00% of total model gain and FRED features 0.00%. Read the WAPE deltas above against those shares before concluding that external data improves point forecasts.

**Share of model gain by feature family**

| Family | Share of gain |
| --- | --- |
| Demand history | 87.79% |
| Product / Store identity | 9.17% |
| Calendar & Events | 1.87% |
| Price | 1.17% |

Top features by gain: `roll_mean_56`, `roll_mean_28`, `item_id`, `roll_std_56`, `roll_mean_7`, `wday`.

**Selection.** smallest feature set within 0.3% relative of the best mean WAPE at h=28. Selected feature set **B** (+ Calendar & Price).

**Objective check** (development mode, fold F1, arm B, then frozen):

| Objective | WAPE | RMSSE | Bias |
| --- | --- | --- | --- |
| tweedie | 0.9426 | 0.7525 | 0.0451 |
| poisson | 0.9484 | 0.7564 | 0.0421 |
| regression | 0.9498 | 0.7602 | 0.0398 |

## Demand shock detection

Scored window: **2016-02-01 to 2016-05-22** (112 days), covering 21,900 eligible series and 2,452,463 scored series-days. Residuals come from feature set B, which contains no FEMA or FRED features by design.

**53,774 episodes** were filed.

| Classification | Episodes | Median duration | Mean peak score |
| --- | --- | --- | --- |
| Volatility Shock | 22095 | 3 days | 50.6 |
| Demand Surge | 17523 | 3 days | 55.3 |
| Demand Collapse | 11872 | 4 days | 51.9 |
| Persistent Over-forecast | 1617 | 5 days | 48.1 |
| Regime Shift | 389 | 35 days | 75.6 |
| Possible Regime Shift (window truncated) | 209 | 33 days | 76.7 |
| Persistent Under-forecast | 69 | 6 days | 53.2 |

| Severity | Episodes |
| --- | --- |
| Critical | 260 |
| Severe | 3234 |
| Elevated | 26805 |
| Watch | 23475 |

**13.8%** of episodes coincided with a FEMA declaration active in the same state. This is a coincidence rate measured over the scored window - it is not evidence of causation, and the platform never presents it as such.

## Business impact (measured, not assumed)

| Quantity | Value |
| --- | --- |
| Series covered | 30,490 (30,463 eligible for planning arithmetic) |
| Actual units sold in the holdout window | 1,231,648 |
| Revenue represented at real M5 prices | $3,900,555 |
| Under-forecast exposure | $1,702,722 |
| Over-forecast exposure | $1,381,154 |
| Total estimated revenue exposure | $3,083,876 |

Estimated revenue exposure dollarises forecast error using real M5 sell prices. It is **not** measured lost revenue: M5 records units sold, so demand that was never satisfied is unobservable in the source data. Safety stock, reorder points and days of cover are shown in the application only after a user supplies lead time and service level, because M5 contains no inventory records at all.

## Reproducing these numbers

```bash
python scripts/run_pipeline.py --mode full
python scripts/export_results.py
```

Library versions: python 3.14.4, pandas 3.0.2, numpy 2.4.4, lightgbm 4.6.0, pyarrow 23.0.1, scipy 1.17.1, scikit-learn 1.8.0.
