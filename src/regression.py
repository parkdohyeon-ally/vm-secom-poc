"""Regression Virtual Metrology — predict a continuous metrology value
(e.g. film thickness) from sensor signals.

The synthetic generator is for METHOD demonstration only. Swap
`make_synthetic_vm` with your real (process-condition -> thickness/CD)
data — e.g. XPS / ellipsometry / deposition logs — to make it authentic.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_validate


class PLSReg(PLSRegression):
    """PLS regression with 1-D predict output (classic VM baseline)."""

    def predict(self, X):
        return super().predict(X).ravel()


def make_synthetic_vm(n=1200, n_sensors=40, noise=0.8, seed=42):
    """Synthetic deposition process: sensor signals -> film thickness (nm)."""
    rng = np.random.default_rng(seed)
    S = rng.normal(0, 1, size=(n, n_sensors))
    beta = np.zeros(n_sensors)
    beta[:5] = [6.0, -3.5, 4.2, 2.1, -1.8]        # dominant process drivers
    nonlin = 1.5 * np.sin(S[:, 5]) + 0.8 * S[:, 6] * S[:, 7]
    thickness = 120 + S @ beta + nonlin + rng.normal(0, noise, size=n)
    X = pd.DataFrame(S, columns=[f"s{i:02d}" for i in range(n_sensors)])
    X = X.mask(rng.random(X.shape) < 0.02)         # mimic fab missingness
    return X, pd.Series(thickness, name="thickness_nm")


def vm_pipeline(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("reg", model),
    ])


def get_regressors():
    return {
        "pls": PLSReg(n_components=8),
        "rf": RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=42),
        "gbm": GradientBoostingRegressor(random_state=42),
    }


def evaluate(X, y, cv=5):
    rows = []
    for name, m in get_regressors().items():
        sc = cross_validate(
            vm_pipeline(m), X, y, cv=cv,
            scoring=["neg_root_mean_squared_error",
                     "neg_mean_absolute_error", "r2"],
        )
        rows.append({
            "model": name,
            "RMSE": round(-sc["test_neg_root_mean_squared_error"].mean(), 3),
            "MAE": round(-sc["test_neg_mean_absolute_error"].mean(), 3),
            "R2": round(sc["test_r2"].mean(), 3),
        })
    return pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    X, y = make_synthetic_vm()
    print(evaluate(X, y).to_string(index=False))
