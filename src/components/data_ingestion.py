"""Read, validate, sort and time-split the source dataset."""
from pathlib import Path
import pandas as pd
from src.logger import get_logger
from src.exception import CustomException
from src.pipeline.predict_pipeline import build_features

logger = get_logger(__name__)
REQUIRED_COLUMNS = ["datetime", "season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed", "casual", "registered", "count"]

class DataIngestion:
    def __init__(self, root=None):
        self.root = Path(root) if root else Path(__file__).resolve().parents[2]

    def run(self):
        try:
            source = self.root / "data/raw/train.csv"
            output = self.root / "data/processed"
            output.mkdir(parents=True, exist_ok=True)
            data = pd.read_csv(source)
            missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            data["datetime"] = pd.to_datetime(data["datetime"], errors="raise")
            data = data.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)
            if data[REQUIRED_COLUMNS].isnull().any().any():
                raise ValueError("Source dataset contains null values in required columns.")
            split_at = int(len(data) * 0.80)
            train, test = data.iloc[:split_at].copy(), data.iloc[split_at:].copy()
            train.to_csv(output / "train.csv", index=False)
            test.to_csv(output / "test.csv", index=False)
            build_features(train).to_csv(output / "X_train_preview.csv", index=False)
            logger.info("Ingestion complete: %s train rows, %s test rows", len(train), len(test))
            return train, test
        except Exception as exc:
            logger.exception("Data ingestion failed")
            raise CustomException(f"Data ingestion failed: {exc}") from exc
