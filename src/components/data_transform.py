"""Build and save leakage-safe preprocessing transformers."""
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from src.logger import get_logger
from src.exception import CustomException
from src.pipeline.predict_pipeline import CATEGORICAL_FEATURES, WEATHER_FEATURES, CALENDAR_NUMERIC_FEATURES

logger = get_logger(__name__)

class DataTransform:
    def __init__(self, root=None):
        self.root = Path(root) if root else Path(__file__).resolve().parents[2]

    def build_preprocessor(self):
        try:
            weather_pipe = Pipeline([("polynomial", PolynomialFeatures(degree=2, include_bias=False)), ("scaler", StandardScaler())])
            calendar_pipe = Pipeline([("scaler", StandardScaler())])
            return ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES), ("weather_polynomial", weather_pipe, WEATHER_FEATURES), ("calendar_numeric", calendar_pipe, CALENDAR_NUMERIC_FEATURES)], remainder="drop")
        except Exception as exc:
            logger.exception("Preprocessor creation failed")
            raise CustomException(f"Preprocessor creation failed: {exc}") from exc

    def run(self):
        try:
            preprocessor = self.build_preprocessor()
            models_dir = self.root / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(preprocessor, models_dir / "preprocessor_template.joblib", compress=3)
            logger.info("Preprocessor template saved; each model Pipeline fits its own clone to avoid leakage.")
            return preprocessor
        except Exception as exc:
            logger.exception("Data transformation setup failed")
            raise CustomException(f"Data transformation setup failed: {exc}") from exc
