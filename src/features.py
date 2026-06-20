"""Leakage-safe preprocessing pipeline.

Everything here is fit inside each CV fold (on train only) when wrapped in a
sklearn Pipeline, so no statistic leaks from validation data.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


class DropHighMissing(BaseEstimator, TransformerMixin):
    """Drop columns whose missing fraction (on train) exceeds `thresh`."""

    def __init__(self, thresh=0.5):
        self.thresh = thresh

    def fit(self, X, y=None):
        Xa = np.asarray(X, dtype=float)
        self.keep_ = np.isnan(Xa).mean(axis=0) < self.thresh
        return self

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        return Xa[:, self.keep_]


def build_preprocessor(missing_thresh=0.5, var_thresh=1e-8):
    return Pipeline([
        ("drop_missing", DropHighMissing(missing_thresh)),
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(var_thresh)),   # drop constant/near-constant
        ("scale", StandardScaler()),
    ])
