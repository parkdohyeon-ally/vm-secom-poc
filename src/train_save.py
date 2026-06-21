"""Train the yield classifier on full SECOM and save artifact for the app."""
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline

from src.data import load_secom
from src.features import build_preprocessor
from src.models import get_models

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models"


def main():
    X, y, _ = load_secom()
    pipe = Pipeline([("pre", build_preprocessor()), ("clf", get_models(y)["rf"])])
    pipe.fit(X, y)
    OUT.mkdir(exist_ok=True)
    joblib.dump({"pipe": pipe, "columns": list(X.columns)}, OUT / "vm_model.joblib")
    print(f"saved -> {OUT/'vm_model.joblib'}")


if __name__ == "__main__":
    main()
