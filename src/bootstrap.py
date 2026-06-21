"""Make the app self-sufficient: download data + train model if missing.

Used by app.py at startup and by the Dockerfile at build time, so a fresh
deploy (HuggingFace Space, container) works with no manual steps.
"""
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODEL = ROOT / "models" / "vm_model.joblib"
BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom"
FILES = ["secom.data", "secom_labels.data"]


def ensure_data():
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = RAW / name
        if not dst.exists():
            urllib.request.urlretrieve(f"{BASE}/{name}", dst)
    return RAW


def ensure_model():
    if not MODEL.exists():
        from src.train_save import main as train
        train()
    return MODEL


def ensure_ready():
    ensure_data()
    ensure_model()


if __name__ == "__main__":
    ensure_ready()
    print("ready:", MODEL.exists())
