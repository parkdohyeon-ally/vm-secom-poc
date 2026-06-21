"""WM-811K wafer-map defect classification (real fabs, 811,457 maps).

9 classes: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random,
Scratch, none. Wafer maps are variable-size 2-D arrays of {0=bg, 1=die,
2=defect}; we resize to a fixed grid for a CNN.

Download (Kaggle):
    kaggle datasets download -d qingyi/wm811k-wafer-map -p data/raw/wm811k
    # -> data/raw/wm811k/LSWMD.pkl  (~3.5 GB)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "wm811k"
LABELS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
          "Near-full", "Random", "Scratch", "none"]


def _norm_label(x):
    """failureType is stored inconsistently (nested arrays / str / empty)."""
    if isinstance(x, np.ndarray):
        return str(x.flatten()[0]) if x.size else ""
    if isinstance(x, (list, tuple)):
        return _norm_label(np.array(x, dtype=object))
    return str(x)


def _resize_map(m, size):
    from skimage.transform import resize
    m = np.asarray(m, dtype=float)
    r = resize(m, (size, size), order=0, preserve_range=True, anti_aliasing=False)
    return (r / 2.0).astype("float32")          # {0,1,2} -> {0,.5,1}


def load_wm811k(size=64, labeled_only=True, max_per_class=None, seed=42):
    pkl = RAW / "LSWMD.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"Put LSWMD.pkl in {RAW} (see module docstring)")
    df = pd.read_pickle(pkl)
    df["label"] = df["failureType"].apply(_norm_label)
    if labeled_only:
        df = df[df["label"].isin(LABELS)]
    if max_per_class:
        df = (df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=seed)))
    X = np.stack([_resize_map(m, size) for m in df["waferMap"]])[:, None, :, :]  # (N,1,H,W)
    y = df["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    return X, y, LABELS
