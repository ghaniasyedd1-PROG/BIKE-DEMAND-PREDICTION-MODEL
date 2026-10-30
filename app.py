"""FastAPI service for bike rental demand predictions."""
from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from src.pipeline.predict_pipeline import build_features
from src.logger import get_logger

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
STATIC = ROOT / "static"
logger = get_logger(__name__)
app = FastAPI(title="Bike Demand Predictor", version="2.0")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def read_json(path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default

metrics = read_json(MODELS / "metrics.json", {})
comparison = read_json(MODELS / "model_comparison.json", {"regression": [], "classification": []})

class BikeInput(BaseModel):
    datetime: str = Field(..., description="Local date/time, e.g. 2011-01-01T08:00")
    season: int = Field(..., ge=1, le=4)
    holiday: int = Field(..., ge=0, le=1)
    workingday: int = Field(..., ge=0, le=1)
    weather: int = Field(..., ge=1, le=4)
    temp: float = Field(..., ge=-10, le=50)
    atemp: float = Field(..., ge=-10, le=60)
    humidity: float = Field(..., ge=0, le=100)
    windspeed: float = Field(..., ge=0, le=100)
    model: str | None = Field(default=None, description="Optional model key, such as svr or random_forest")


def model_names(task):
    return sorted(p.stem[len(task)+1:] for p in MODELS.glob(f"{task}_*.joblib"))


def choose_model(task, requested):
    available = model_names(task)
    if not available:
        raise HTTPException(status_code=503, detail="Models are not trained yet. Run: make train")
    if requested:
        key = requested.strip().lower().replace(" ", "_").replace("-", "_")
        if key not in available:
            raise HTTPException(status_code=400, detail=f"Unknown model '{requested}'. Available models: {available}")
        return key
    best = metrics.get("regression", {}).get("best_model") if task == "regression" else metrics.get("classification", {}).get("best_model")
    return best if best in available else available[0]

@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(model_names("regression") and model_names("classification")), "regression_models": model_names("regression"), "classification_models": model_names("classification")}

@app.get("/api/metrics")
def get_metrics():
    return metrics

@app.get("/api/model-comparison")
def get_model_comparison():
    return comparison

@app.post("/predict/regression")
def predict_regression(payload: BikeInput):
    try:
        selected = choose_model("regression", payload.model)
        values = payload.model_dump(exclude={"model"})
        X = build_features(pd.DataFrame([values]))
        bundle = joblib.load(MODELS / f"regression_{selected}.joblib")
        prediction = max(0.0, float(bundle["model"].predict(X)[0]))
        return {"prediction": round(prediction), "unit": "bikes", "model": selected}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Regression API failed")
        raise HTTPException(status_code=400, detail=f"Could not make prediction: {exc}") from exc

@app.post("/predict/classification")
def predict_classification(payload: BikeInput):
    try:
        selected = choose_model("classification", payload.model)
        values = payload.model_dump(exclude={"model"})
        X = build_features(pd.DataFrame([values]))
        bundle = joblib.load(MODELS / f"classification_{selected}.joblib")
        prediction = int(bundle["model"].predict(X)[0])
        label = {0: "Low", 1: "Medium", 2: "High"}[prediction]
        confidence = None
        if hasattr(bundle["model"], "predict_proba"):
            confidence = round(float(max(bundle["model"].predict_proba(X)[0])), 4)
        return {"prediction": label, "class_id": prediction, "confidence": confidence, "model": selected, "meaning": {"Low": "Expected demand is in the lower range.", "Medium": "Expected demand is in the middle range.", "High": "Expected demand is in the higher range."}[label]}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Classification API failed")
        raise HTTPException(status_code=400, detail=f"Could not classify demand: {exc}") from exc
