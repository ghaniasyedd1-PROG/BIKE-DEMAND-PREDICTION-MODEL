"""Shared feature engineering and prediction functions for training and serving."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)
RAW_FEATURES = ["season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed"]
CATEGORICAL_FEATURES = ["season", "holiday", "workingday", "weather", "hour_workingday", "hour_season", "hour_work_season"]
WEATHER_FEATURES = ["temp", "atemp", "humidity", "windspeed"]
CALENDAR_NUMERIC_FEATURES = ["year", "month", "hour", "day_of_week", "is_weekend"]
NUMERIC_FEATURES = WEATHER_FEATURES + CALENDAR_NUMERIC_FEATURES
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create model features in one place; never use casual or registered counts."""
    try:
        frame = data.copy()
        if "datetime" not in frame.columns:
            raise ValueError("Input data must contain a datetime column.")
        dt = pd.to_datetime(frame["datetime"], errors="raise")
        for col in RAW_FEATURES:
            if col not in frame.columns:
                raise ValueError(f"Input data is missing required feature: {col}")
        frame["year"] = dt.dt.year
        frame["month"] = dt.dt.month
        frame["hour"] = dt.dt.hour
        frame["day_of_week"] = dt.dt.dayofweek
        frame["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        frame["hour_workingday"] = dt.dt.hour.astype(str) + "_" + frame["workingday"].astype(str)
        frame["hour_season"] = dt.dt.hour.astype(str) + "_" + frame["season"].astype(str)
        frame["hour_work_season"] = (dt.dt.hour.astype(str) + "_" + frame["workingday"].astype(str) + "_" + frame["season"].astype(str))
        # Keep only approved predictors. day/day_of_year and rider-subtotal columns are excluded.
        return frame[FEATURES].copy()
    except Exception as exc:
        logger.exception("Feature building failed")
        raise CustomException(f"Feature building failed: {exc}") from exc


def predict(model_name: str, payload: dict, task: str = "regression"):
    """Load a saved model and predict one request row."""
    try:
        root = Path(__file__).resolve().parents[2]
        path = root / "models" / f"{task}_{model_name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path.name}")
        bundle = joblib.load(path)
        row = pd.DataFrame([payload])
        features = build_features(row)
        model = bundle["model"]
        result = model.predict(features)[0]
        if task == "regression":
            return max(0.0, float(result)), bundle
        label_map = {0: "Low", 1: "Medium", 2: "High"}
        label = label_map[int(result)]
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(features)[0]))
        return label, bundle, confidence
    except Exception as exc:
        logger.exception("Prediction failed")
        raise CustomException(f"Prediction failed: {exc}") from exc
