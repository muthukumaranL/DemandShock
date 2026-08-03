"""Dependency smoke test.

Python 3.14 / pandas 3.0 are new enough that the pipeline must not assume any
API behaves the way older recipes expect. This script exercises every library
call the pipeline actually relies on, before any design decision hardens.

Run:  python scripts/smoke_deps.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        try:
            detail = fn() or ""
            RESULTS.append((name, True, str(detail)))
        except Exception as exc:  # noqa: BLE001 - smoke test reports, never raises
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
        return fn

    return deco


@check("python")
def _python():
    assert sys.version_info >= (3, 11)
    return sys.version.split()[0]


@check("numpy: ravel/repeat/tile melt primitives")
def _numpy():
    import numpy as np

    mat = np.arange(6, dtype=np.int16).reshape(2, 3)
    flat = mat.ravel(order="C")
    assert flat.tolist() == [0, 1, 2, 3, 4, 5]
    assert np.repeat(np.array([0, 1], dtype=np.int8), 3).tolist() == [0, 0, 0, 1, 1, 1]
    assert np.tile(np.arange(1, 4, dtype=np.int16), 2).tolist() == [1, 2, 3, 1, 2, 3]
    return np.__version__


@check("pandas: Categorical.from_codes")
def _cat_from_codes():
    import numpy as np
    import pandas as pd

    cat = pd.Categorical.from_codes(
        np.array([0, 1, 0], dtype=np.int8), categories=["CA_1", "CA_2"]
    )
    assert list(cat) == ["CA_1", "CA_2", "CA_1"]
    return pd.__version__


@check("pandas: groupby shift + rolling (leakage-critical)")
def _groupby_rolling():
    import pandas as pd

    df = pd.DataFrame({"id": ["a"] * 6, "y": [1.0, 2, 3, 4, 5, 6]})
    shifted = df.groupby("id", observed=True)["y"].shift(2)
    rolled = shifted.groupby(df["id"], observed=True).rolling(2).mean().reset_index(drop=True)
    # rows 0-2 are NaN (shift eats 2, rolling eats 1 more); row 3 = mean(1,2)
    assert rolled.isna().tolist()[:3] == [True, True, True]
    assert abs(rolled.iloc[3] - 1.5) < 1e-9
    return "shift-before-rolling verified"


@check("pandas: merge_asof (FRED availability join)")
def _merge_asof():
    import pandas as pd

    left = pd.DataFrame({"date": pd.to_datetime(["2016-04-19", "2016-04-20"])})
    right = pd.DataFrame(
        {
            "available_from": pd.to_datetime(["2016-03-20", "2016-04-20"]),
            "ur": [5.4, 5.3],
        }
    )
    out = pd.merge_asof(left, right, left_on="date", right_on="available_from")
    assert out["ur"].tolist() == [5.4, 5.3]
    return "as-of boundary verified"


@check("pandas: merge validate='m:1'")
def _merge_validate():
    import pandas as pd

    left = pd.DataFrame({"k": [1, 1, 2], "v": [1, 2, 3]})
    right = pd.DataFrame({"k": [1, 2], "w": ["a", "b"]})
    out = left.merge(right, on="k", how="left", validate="m:1")
    assert len(out) == 3
    try:
        left.merge(pd.DataFrame({"k": [1, 1], "w": ["a", "b"]}), on="k", validate="m:1")
    except Exception:
        return "m:1 validation enforced"
    raise AssertionError("m:1 validation did not raise on a duplicated key")


@check("pyarrow: partitioned dataset round-trip")
def _pyarrow():
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds

    with tempfile.TemporaryDirectory() as tmp:
        df = pd.DataFrame(
            {
                "store_id": pd.Categorical(["CA_1", "CA_1", "TX_1"]),
                "d": pd.array([1, 2, 1], dtype="int16"),
                "sales": pd.array([3, 0, 7], dtype="int16"),
            }
        )
        ds.write_dataset(
            pa.Table.from_pandas(df, preserve_index=False),
            base_dir=tmp,
            format="parquet",
            partitioning=ds.partitioning(
                pa.schema([("store_id", pa.string())]), flavor="hive"
            ),
            existing_data_behavior="overwrite_or_ignore",
        )
        back = ds.dataset(tmp, format="parquet", partitioning="hive").to_table(
            columns=["store_id", "d", "sales"],
            filter=ds.field("store_id") == "CA_1",
        ).to_pandas()
        assert len(back) == 2 and back["sales"].sum() == 3
        return f"pyarrow {pa.__version__}, filter pushdown + column pruning OK"


@check("lightgbm: train / predict / pred_contrib")
def _lightgbm():
    import lightgbm as lgb
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.random((400, 4), dtype=np.float32)
    y = (x[:, 0] * 5).astype(np.float32)
    dtrain = lgb.Dataset(x, label=y, free_raw_data=False)
    booster = lgb.train(
        {"objective": "tweedie", "tweedie_variance_power": 1.1, "num_leaves": 8,
         "verbose": -1, "seed": 42, "min_data_in_leaf": 5},
        dtrain,
        num_boost_round=15,
    )
    pred = booster.predict(x[:5])
    contrib = booster.predict(x[:5], pred_contrib=True)
    assert pred.shape == (5,)
    assert contrib.shape == (5, 5), contrib.shape  # n_features + base value
    # exact TreeSHAP identity: contributions + base == raw score (log link -> exp)
    assert np.allclose(np.exp(contrib.sum(axis=1)), pred, rtol=1e-5)
    return f"lightgbm {lgb.__version__}, pred_contrib == exact TreeSHAP"


@check("lightgbm: early stopping callback")
def _lightgbm_es():
    import lightgbm as lgb
    import numpy as np

    rng = np.random.default_rng(1)
    x = rng.random((300, 3), dtype=np.float32)
    y = (x[:, 0] * 3).astype(np.float32)
    dtrain = lgb.Dataset(x[:200], label=y[:200])
    dvalid = lgb.Dataset(x[200:], label=y[200:], reference=dtrain)
    booster = lgb.train(
        {"objective": "regression", "num_leaves": 8, "verbose": -1, "seed": 42,
         "min_data_in_leaf": 5, "metric": "rmse"},
        dtrain,
        num_boost_round=200,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(10, verbose=False)],
    )
    assert booster.best_iteration > 0
    return f"best_iteration={booster.best_iteration}"


@check("lightgbm: pandas categorical features")
def _lightgbm_cat():
    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    n = 300
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "store_id": pd.Categorical(rng.choice(["CA_1", "CA_2", "TX_1"], n)),
            "x": rng.random(n).astype(np.float32),
        }
    )
    y = df["x"].to_numpy() * 2
    booster = lgb.train(
        {"objective": "regression", "num_leaves": 8, "verbose": -1, "seed": 42,
         "min_data_in_leaf": 5},
        lgb.Dataset(df, label=y, categorical_feature=["store_id"]),
        num_boost_round=10,
    )
    assert booster.predict(df.head(3)).shape == (3,)
    return "native categorical handling OK"


@check("scipy: norm.ppf (service level -> z)")
def _scipy():
    import scipy
    from scipy.stats import norm

    assert abs(float(norm.ppf(0.95)) - 1.6448536) < 1e-6
    return scipy.__version__


@check("sklearn / joblib")
def _sklearn():
    import joblib
    import sklearn

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "o.joblib"
        joblib.dump({"a": 1}, p)
        assert joblib.load(p) == {"a": 1}
    return f"sklearn {sklearn.__version__}, joblib {joblib.__version__}"


@check("streamlit / plotly / fastapi imports")
def _app_stack():
    import fastapi
    import plotly
    import plotly.graph_objects as go
    import streamlit

    go.Figure()
    return (
        f"streamlit {streamlit.__version__}, plotly {plotly.__version__}, "
        f"fastapi {fastapi.__version__}"
    )


@check("fastapi TestClient (needs httpx)")
def _testclient():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
    return "TestClient OK"


@check("yaml config load")
def _yaml():
    import yaml

    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    assert cfg["project"]["name"] == "DemandShock"
    return f"PyYAML {yaml.__version__}"


@check("shap (optional)")
def _shap():
    try:
        import shap  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"NOT INSTALLED ({type(exc).__name__}) - using LightGBM native TreeSHAP"
    return f"shap {shap.__version__} available (optional path)"


def main() -> int:
    width = max(len(n) for n, _, _ in RESULTS) + 2
    print("\n=== DemandShock dependency smoke test ===")
    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}} {detail}")
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed.")
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    print("All required dependencies behave as the pipeline expects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
