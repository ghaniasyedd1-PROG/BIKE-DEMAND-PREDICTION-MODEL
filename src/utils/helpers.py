"""Shared small helpers for data and model files."""
import json
from pathlib import Path
import joblib


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_joblib(path):
    return joblib.load(path)
