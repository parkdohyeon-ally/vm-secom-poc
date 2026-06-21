"""Imbalanced-aware, leakage-free cross-validation reporting."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    matthews_corrcoef,
)


def get_scores(model, X):
    """Continuous score for the positive (fail) class."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.asarray(model.decision_function(X)).ravel()


def recall_at_precision(y, scores, min_precision=0.5):
    """Best achievable recall while precision >= min_precision."""
    p, r, _ = precision_recall_curve(y, scores)
    mask = p[:-1] >= min_precision
    return float(r[:-1][mask].max()) if mask.any() else 0.0


def best_mcc(y, scores, n_grid=200):
    """Max Matthews correlation coefficient over thresholds (Bosch's metric).

    MCC needs hard labels, so we sweep candidate thresholds and report the
    best. Returns (mcc, threshold).
    """
    grid = np.unique(np.quantile(scores, np.linspace(0, 1, n_grid)))
    best_m, best_t = -1.0, 0.5
    for t in grid:
        m = matthews_corrcoef(y, (scores >= t).astype(int))
        if m > best_m:
            best_m, best_t = m, t
    return float(best_m), float(best_t)


def cross_val_report(name, estimator, X, y, n_splits=5, seed=42, min_precision=0.5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for tr, te in skf.split(X, y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        m = clone(estimator)
        m.fit(Xtr, ytr)
        s = get_scores(m, Xte)
        mcc, _ = best_mcc(yte, s)
        rows.append({
            "pr_auc": average_precision_score(yte, s),
            "roc_auc": roc_auc_score(yte, s),
            "mcc": mcc,
            f"recall@p{int(min_precision*100)}": recall_at_precision(yte, s, min_precision),
        })
    mean = pd.DataFrame(rows).mean().round(4).to_dict()
    return {"model": name, **mean}
