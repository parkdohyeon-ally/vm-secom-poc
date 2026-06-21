"""Bosch Production Line Performance — large-scale tabular quality prediction.

Real anonymized manufacturing line: ~1.18M parts x ~968 numeric features,
extreme imbalance (~0.58% fail). Same VM/FDC structure as SECOM, 1000x scale.

Download (needs Kaggle account + competition rules accepted):
    pip install kaggle
    kaggle competitions download -c bosch-production-line-performance \
        -f train_numeric.csv.zip -p data/raw
    # unzip -> data/raw/train_numeric.csv  (~2 GB)

The full numeric CSV is too large to load at once on most laptops — use the
sampling / chunked loaders below.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
NUMERIC = RAW / "train_numeric.csv"


def load_bosch_sample(n_rows=100_000, target="Response"):
    """Quick start: first n_rows (id + numeric features + Response label)."""
    if not NUMERIC.exists():
        raise FileNotFoundError("See module docstring to download train_numeric.csv")
    df = pd.read_csv(NUMERIC, nrows=n_rows)
    y = df[target].astype(int).rename("fail")
    X = df.drop(columns=["Id", target], errors="ignore")
    return X, y


def iter_bosch_chunks(chunksize=200_000, target="Response"):
    """Stream the full file in chunks (for out-of-core training/aggregation)."""
    if not NUMERIC.exists():
        raise FileNotFoundError("See module docstring to download train_numeric.csv")
    for chunk in pd.read_csv(NUMERIC, chunksize=chunksize):
        y = chunk[target].astype(int).rename("fail")
        X = chunk.drop(columns=["Id", target], errors="ignore")
        yield X, y
