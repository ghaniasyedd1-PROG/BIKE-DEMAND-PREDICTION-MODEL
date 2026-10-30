"""Run the complete ingestion -> transformation -> model training workflow."""
from src.components.data_ingestion import DataIngestion
from src.components.data_transform import DataTransform
from src.components.model_trainer import ModelTrainer
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

def main():
    try:
        train, test = DataIngestion().run()
        DataTransform().run()
        results = ModelTrainer().run(train, test)
        logger.info("Training pipeline finished successfully.")
        print("Training complete. Best regression:", results["regression"]["best_model"])
        print("Best classification:", results["classification"]["best_model"])
        print("See models/metrics.json and models/model_comparison.json for results.")
    except Exception as exc:
        logger.exception("Training pipeline failed")
        raise CustomException(f"Training pipeline failed: {exc}") from exc

if __name__ == "__main__":
    main()
