from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from src.pipeline.predict_pipeline import build_features

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

@pytest.mark.parametrize("task,name", [("regression","random_forest"),("classification","random_forest")])
def test_model_can_load_and_predict(task, name):
    path = MODELS / f"{task}_{name}.joblib"
    if not path.exists():
        pytest.skip("Run `make train` to create model artifacts first")
    bundle = joblib.load(path)
    row = {"datetime":"2012-06-15T08:00:00", "season":2, "holiday":0, "workingday":1, "weather":1, "temp":25.0, "atemp":27.0, "humidity":60, "windspeed":10}
    X = build_features(pd.DataFrame([row]))
    prediction = bundle["model"].predict(X)[0]
    assert np.isfinite(float(prediction))
    if task == "regression": assert prediction >= 0

def test_transformed_features_have_no_nulls():
    path = MODELS / "regression_random_forest.joblib"
    if not path.exists(): pytest.skip("Run `make train` first")
    bundle = joblib.load(path)
    row = {"datetime":"2012-06-15T08:00:00", "season":2, "holiday":0, "workingday":1, "weather":1, "temp":25.0, "atemp":27.0, "humidity":60, "windspeed":10}
    transformed = bundle["model"].named_steps["preprocessor"].transform(build_features(pd.DataFrame([row])))
    values = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    assert np.isfinite(values).all()

def test_best_models_meet_requested_quality_thresholds():
    path = MODELS / "metrics.json"
    if not path.exists(): pytest.skip("Run `make train` first")
    metrics = json.loads(path.read_text())
    best_reg = metrics["regression"]["models"][metrics["regression"]["best_model"]]["R2"]
    best_clf = metrics["classification"]["models"][metrics["classification"]["best_model"]]["accuracy"]
    assert best_reg >= 0.8, f"Best regression R2={best_reg:.3f}, required >= 0.8"
    assert best_clf >= 0.8, f"Best classifier accuracy={best_clf:.3f}, required >= 0.8"
