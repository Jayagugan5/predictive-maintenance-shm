"""Serving layer.  Run:  uvicorn api:app --reload   ->  POST /predict {"features": {...15 features...}}"""
from pathlib import Path
import joblib, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

bundle = joblib.load(Path(__file__).parent / "outputs" / "model.joblib")
app = FastAPI(title="Structural risk inference")


class Reading(BaseModel):
    features: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "features": bundle["features"]}


@app.post("/predict")
def predict(r: Reading):
    missing = [f for f in bundle["features"] if f not in r.features]
    if missing:
        raise HTTPException(422, f"missing features: {missing}")
    x = np.array([[r.features[f] for f in bundle["features"]]])
    p = bundle["model"].predict_proba(x)[0]
    return {"risk_class": bundle["classes"][int(p.argmax())],
            "probabilities": dict(zip(bundle["classes"], map(float, p))),
            "risk_score": float(0.5 * p[1] + p[2])}
