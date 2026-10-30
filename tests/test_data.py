from pathlib import Path
import pandas as pd
from src.components.data_ingestion import REQUIRED_COLUMNS
from src.pipeline.predict_pipeline import build_features

ROOT = Path(__file__).resolve().parents[1]

def test_dataset_has_required_columns():
    if not (ROOT / "data/raw/train.csv").exists():
        import pytest
        pytest.skip("Download Kaggle train.csv into data/raw/ to run dataset checks")
    data = pd.read_csv(ROOT / "data/raw/train.csv")
    assert set(REQUIRED_COLUMNS).issubset(data.columns)

def test_shared_feature_builder_returns_expected_columns():
    row = {"datetime":"2012-06-15T08:00:00", "season":2, "holiday":0, "workingday":1, "weather":1, "temp":25.0, "atemp":27.0, "humidity":60, "windspeed":10}
    result = build_features(pd.DataFrame([row]))
    assert not result.isnull().any().any()
    assert "casual" not in result.columns and "registered" not in result.columns
    assert "day" not in result.columns and "day_of_year" not in result.columns

def test_time_split_is_chronological():
    if not (ROOT / "data/processed/train.csv").exists() or not (ROOT / "data/processed/test.csv").exists():
        import pytest
        pytest.skip("Run make train to create the processed time split")
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    test = pd.read_csv(ROOT / "data/processed/test.csv")
    assert pd.to_datetime(train.datetime).max() < pd.to_datetime(test.datetime).min()
