"""Offline smoke tests — no network/data download needed (CI-friendly)."""
import numpy as np
import pandas as pd

from src.roi import roi
from src.regression import make_synthetic_vm, evaluate
from src.models import get_models
from src.features import build_preprocessor
from src.evaluate import best_mcc


def test_roi_positive():
    r = roi(50_000, 400, 0.066, 0.40, inspection_cost_per_wafer=0.02)
    assert r["net_saved_usd"] > 0
    assert {"fails_per_month", "net_saved_usd"}.issubset(r)


def test_regression_vm_fits():
    X, y = make_synthetic_vm(n=400, n_sensors=20)
    df = evaluate(X, y, cv=3)
    assert df["R2"].max() > 0.5


def test_models_present():
    y = pd.Series([0, 1, 0, 1, 0, 1])
    assert {"logreg", "pls", "rf"}.issubset(get_models(y))


def test_preprocessor_drops_constant():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "const": [5, 5, 5, 5],
                      "b": [1.0, np.nan, 3.0, 4.0]})
    Xt = build_preprocessor().fit_transform(X)
    assert Xt.shape[0] == 4 and Xt.shape[1] < X.shape[1]


def test_best_mcc_range():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    mcc, thr = best_mcc(y, s)
    assert -1.0 <= mcc <= 1.0 and mcc > 0
