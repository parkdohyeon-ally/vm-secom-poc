"""End-to-end baseline: load -> preprocess -> CV-compare models -> save table.

Usage:  python run_baseline.py
"""
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import load_secom
from src.features import build_preprocessor
from src.models import get_models
from src.evaluate import cross_val_report

ROOT = Path(__file__).resolve().parent


def main():
    X, y, ts = load_secom()
    print(f"Loaded X={X.shape}  fail_rate={y.mean():.4f}  "
          f"({int(y.sum())} fails / {len(y)})")

    results = []
    for name, clf in get_models(y).items():
        pipe = Pipeline([("pre", build_preprocessor()), ("clf", clf)])
        rep = cross_val_report(name, pipe, X, y, n_splits=5)
        print(f"  {name:8s}  "
              + "  ".join(f"{k}={v}" for k, v in rep.items() if k != "model"))
        results.append(rep)

    out = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    rep_dir = ROOT / "reports"
    rep_dir.mkdir(exist_ok=True)
    out.to_csv(rep_dir / "metrics.csv", index=False)
    print("\n=== ranking (by PR-AUC) ===")
    print(out.to_string(index=False))
    print(f"\nsaved -> {rep_dir/'metrics.csv'}")
    # sanity baseline: a no-skill classifier's PR-AUC ~= fail rate
    print(f"(no-skill PR-AUC reference = base fail rate = {y.mean():.4f})")


if __name__ == "__main__":
    main()
