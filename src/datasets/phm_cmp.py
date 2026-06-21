"""PHM 2016 Data Challenge — CMP (Chemical Mechanical Planarization)
material removal-rate regression. A *real* semiconductor Virtual Metrology
task: predict Avg Removal Rate (MRR) from in-situ sensor traces.

Data: PHM Society 2016 Data Challenge (free, registration).
Place files under data/raw/phm/ — typically per-wafer sensor time-series
plus a removal-rate label table.

Sensor traces are time-series per (wafer, stage); we aggregate each into
summary features (mean/std/min/max) for a tabular VM regressor.
Adjust `id_cols` / `target` to the actual column names of your download.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "phm"


def load_phm_cmp(traces_csv="train.csv", labels_csv="train_labels.csv",
                 id_cols=("WAFER_ID", "STAGE"), target="AVG_REMOVAL_RATE"):
    tp, lp = RAW / traces_csv, RAW / labels_csv
    if not tp.exists():
        raise FileNotFoundError(f"Put PHM CMP files in {RAW} (see module docstring)")

    traces = pd.read_csv(tp)
    labels = pd.read_csv(lp)
    id_cols = list(id_cols)
    sensors = [c for c in traces.columns
               if c not in id_cols and pd.api.types.is_numeric_dtype(traces[c])]

    agg = traces.groupby(id_cols)[sensors].agg(["mean", "std", "min", "max"])
    agg.columns = ["_".join(c) for c in agg.columns]
    df = agg.reset_index().merge(labels[id_cols + [target]], on=id_cols)

    X = df.drop(columns=id_cols + [target])
    y = df[target].rename("removal_rate")
    return X, y
