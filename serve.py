"""FastAPI serving for the VM yield model.

Run:  uvicorn serve:app --reload      (after `python -m src.train_save`)
Docs: http://127.0.0.1:8000/docs
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL = Path(__file__).resolve().parent / "models" / "vm_model.joblib"
app = FastAPI(title="Virtual Metrology API", version="0.1.0")
_artifact = None


def _model():
    global _artifact
    if _artifact is None:
        if not MODEL.exists():
            raise HTTPException(503, "model missing — run `python -m src.train_save`")
        _artifact = joblib.load(MODEL)
    return _artifact


class Wafer(BaseModel):
    """Sensor readings as {sensor_id: value}. Missing sensors -> NaN (imputed)."""
    features: dict


@app.get("/health")
def health():
    return {"status": "ok", "model_present": MODEL.exists()}


@app.post("/predict")
def predict(w: Wafer):
    art = _model()
    cols, pipe = art["columns"], art["pipe"]
    row = pd.DataFrame([{c: w.features.get(c, np.nan) for c in cols}])
    prob = float(pipe.predict_proba(row)[0, 1])
    return {
        "fail_probability": round(prob, 4),
        "verdict": "FAIL" if prob >= 0.5 else "PASS",
        "threshold": 0.5,
    }
