"""Candidate models for SECOM yield prediction.

PLS is the classic Virtual Metrology baseline (latent projection on
collinear sensor data). Tree ensembles handle non-linearity + imbalance.
XGBoost / LightGBM are optional (used if installed).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression


class PLSClassifier(BaseEstimator, ClassifierMixin):
    """PLS regression on {0,1} target, thresholded — classic VM baseline."""

    def __init__(self, n_components=12):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        n = min(self.n_components, np.asarray(X).shape[1])
        self.pls_ = PLSRegression(n_components=max(1, n))
        self.pls_.fit(X, np.asarray(y, dtype=float))
        return self

    def decision_function(self, X):
        return self.pls_.predict(X).ravel()

    def predict(self, X):
        return (self.decision_function(X) > 0.5).astype(int)


def get_models(y):
    pos = int((np.asarray(y) == 1).sum())
    neg = int((np.asarray(y) == 0).sum())
    spw = neg / max(pos, 1)

    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "pls": PLSClassifier(n_components=12),
        "rf": RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["xgb"] = XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6, scale_pos_weight=spw,
            eval_metric="aucpr", n_jobs=-1, random_state=42,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["lgbm"] = LGBMClassifier(
            n_estimators=600, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6, class_weight="balanced",
            n_jobs=-1, random_state=42, verbosity=-1,
        )
    except Exception:
        pass
    return models
