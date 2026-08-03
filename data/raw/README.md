# Raw data

The source files are **not stored in this repository**. The M5 competition data is
subject to Kaggle's terms and is not redistributed here.

`config.yaml` points at the real locations on disk:

```yaml
paths:
  m5_dir: datasets/m5-forecasting-accuracy
  external_dir: datasets
```

## Expected layout

```
datasets/
├── m5-forecasting-accuracy/
│   ├── calendar.csv                 # 1,969 rows, 2011-01-29 .. 2016-06-19
│   ├── sales_train_evaluation.csv   # 30,490 x 1,947 - canonical actuals (d_1..d_1941)
│   ├── sales_train_validation.csv   # cross-check only (d_1..d_1913)
│   └── sell_prices.csv              # 6,841,121 weekly store/item prices
├── DisasterDeclarationsSummaries.csv # OpenFEMA v2 disaster declarations
├── CAUR.csv                          # FRED: California unemployment rate, monthly
├── TXUR.csv                          # FRED: Texas
└── WIUR.csv                          # FRED: Wisconsin
```

## Where to get them

| Source | Link |
| --- | --- |
| M5 Forecasting - Accuracy | https://www.kaggle.com/competitions/m5-forecasting-accuracy/data |
| FEMA disaster declarations | https://www.fema.gov/openfema-data-page/disaster-declarations-summaries-v2 |
| FRED state unemployment | https://fred.stlouisfed.org/series/CAUR (also TXUR, WIUR) |

## Deliberately unused

These files may sit alongside the ones above; the pipeline never reads them and
`config.yaml` lists them under `unused_files`:

- `4365920.csv`, `4365923.csv` — NOAA GHCN-Daily weather extracts. Weather is out of
  scope in this version; it is documented as a possible future extension only.
- `sample_submission.csv` — a Kaggle submission scaffold with no analytical value here.
- `*.xlsx` duplicates of the FRED series.
