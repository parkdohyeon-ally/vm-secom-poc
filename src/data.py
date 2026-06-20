"""Load SECOM features, labels, timestamps."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def load_secom():
    """Return (X, y, ts).

    X  : DataFrame [n_wafers, 590] sensor features (NaN preserved)
    y  : Series, 1 = fail (positive/minority), 0 = pass
    ts : Series of timestamps (for time-based splits)
    """
    feat_path = RAW / "secom.data"
    lab_path = RAW / "secom_labels.data"
    if not feat_path.exists():
        raise FileNotFoundError("Run `python data/get_data.py` first.")

    X = pd.read_csv(feat_path, sep=r"\s+", header=None, na_values="NaN")
    X.columns = [f"f{i:03d}" for i in range(X.shape[1])]

    lab = pd.read_csv(lab_path, sep=r"\s+", header=None)
    # col0 = label {-1 pass, 1 fail}; col1 = date; col2 = time
    y = lab[0].map({-1: 0, 1: 1}).astype(int).rename("fail")
    ts = pd.to_datetime(
        lab[1].astype(str) + " " + lab[2].astype(str),
        format="%d/%m/%Y %H:%M:%S", errors="coerce",
    ).rename("ts")
    return X, y, ts


if __name__ == "__main__":
    X, y, ts = load_secom()
    print("X:", X.shape, "| fail rate:", round(float(y.mean()), 4))
    print("missing per col (mean):", round(float(X.isna().mean().mean()), 4))
    print("time span:", ts.min(), "->", ts.max())
