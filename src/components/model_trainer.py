"""Train, tune, evaluate and save all required regression/classification models."""
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree
from sklearn.naive_bayes import GaussianNB
from src.models.regression.decision_tree import build_model as decision_tree_reg
from src.models.regression.svm import build_model as svr_reg
from src.models.regression.random_forest import build_model as random_forest_reg
from src.models.classification.decision_tree import build_model as decision_tree_clf
from src.models.classification.naive_bayes import build_model as naive_bayes_clf
from src.models.classification.svm import build_model as svm_clf
from src.models.classification.random_forest import build_model as random_forest_clf
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score, confusion_matrix, classification_report
from src.logger import get_logger
from src.exception import CustomException
from src.pipeline.predict_pipeline import build_features
from src.components.data_transform import DataTransform
from src.utils.helpers import save_json

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, root=None):
        self.root = Path(root) if root else Path(__file__).resolve().parents[2]
        self.model_dir = self.root / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cv = TimeSeriesSplit(n_splits=3)
        self.preprocessor = DataTransform(self.root).build_preprocessor()

    def _pipeline(self, estimator):
        preprocessor = self.preprocessor
        if isinstance(estimator, GaussianNB):
            # GaussianNB requires dense arrays; force ColumnTransformer to return dense output.
            preprocessor = DataTransform(self.root).build_preprocessor()
            preprocessor.sparse_threshold = 0.0
        return Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    def _search(self, estimator, params, X, y, scoring):
        search = GridSearchCV(self._pipeline(estimator), params, cv=self.cv, scoring=scoring, n_jobs=1, refit=True, error_score="raise")
        # Use the earliest 4,000 training rows for faster, still chronological tuning.
        tune_size = min(4000, len(X))
        search.fit(X.iloc[:tune_size], y.iloc[:tune_size] if hasattr(y, "iloc") else y[:tune_size])
        logger.info("Tuning %s best params: %s", estimator.__class__.__name__, search.best_params_)
        return search.best_estimator_, search.best_params_

    def run(self, train, test):
        try:
            X_train = build_features(train)
            X_test = build_features(test)
            y_train = train["count"].astype(float)
            y_test = test["count"].astype(float)
            # Demand cut points are calculated only from the training period.
            low_cut, high_cut = [float(v) for v in y_train.quantile([1/3, 2/3]).tolist()]
            yc_train = np.where(y_train <= low_cut, 0, np.where(y_train <= high_cut, 1, 2)).astype(int)
            yc_test = np.where(y_test <= low_cut, 0, np.where(y_test <= high_cut, 1, 2)).astype(int)
            # Each required model is defined in its own file under src/models/.
            reg_specs = {
                "decision_tree": (decision_tree_reg(), {"model__max_depth": [10, None], "model__min_samples_leaf": [1, 3]}),
                "svr": (svr_reg(), {"model__C": [10, 30], "model__epsilon": [0.1], "model__gamma": ["scale"]}),
                "random_forest": (random_forest_reg(), {"model__n_estimators": [120], "model__max_depth": [None, 20], "model__min_samples_leaf": [1, 2]}),
            }
            regression_results = {}
            regression_models = {}
            for name, (estimator, params) in reg_specs.items():
                logger.info("Training regression model: %s", name)
                if params:
                    model, best_params = self._search(estimator, params, X_train, y_train, "r2")
                else:
                    model = Pipeline([("preprocessor", self.preprocessor), ("model", estimator)])
                    model.fit(X_train, y_train)
                    best_params = {"target_transform": "log1p"}
                model.fit(X_train, y_train)
                predictions = np.maximum(0, model.predict(X_test))
                metrics = {"R2": float(r2_score(y_test, predictions)), "MAE": float(mean_absolute_error(y_test, predictions)), "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))), "best_params": best_params}
                regression_results[name] = metrics
                regression_models[name] = model
                joblib.dump({"model": model, "features": list(X_train.columns), "name": name}, self.model_dir / f"regression_{name}.joblib", compress=3)
                logger.info("%s regression results: %s", name, metrics)
                if name == "linear_regression":
                    self._save_actual_plot(y_test, predictions)
            best_reg_name = max(regression_results, key=lambda n: regression_results[n]["R2"])
            # Naive Bayes is intentionally classification-only.
            clf_specs = {
                "decision_tree": (decision_tree_clf(), {"model__max_depth": [10, None], "model__min_samples_leaf": [1, 3]}),
                "naive_bayes": (naive_bayes_clf(), None),
                "svm": (svm_clf(), {"model__C": [1, 10], "model__gamma": ["scale"]}),
                "random_forest": (random_forest_clf(), {"model__n_estimators": [120], "model__max_depth": [None, 20], "model__min_samples_leaf": [1, 2]}),
            }
            classification_results = {}
            classification_models = {}
            for name, (estimator, params) in clf_specs.items():
                logger.info("Training classification model: %s", name)
                if params:
                    model, best_params = self._search(estimator, params, X_train, yc_train, "f1_macro")
                else:
                    model = self._pipeline(estimator)
                    model.fit(X_train, yc_train)
                    best_params = {"default": "scikit-learn estimator defaults"}
                model.fit(X_train, yc_train)
                predictions = model.predict(X_test)
                metrics = {"accuracy": float(accuracy_score(yc_test, predictions)), "macro_f1": float(f1_score(yc_test, predictions, average="macro", zero_division=0)), "confusion_matrix": confusion_matrix(yc_test, predictions, labels=[0,1,2]).tolist(), "best_params": best_params, "classification_report": classification_report(yc_test, predictions, labels=[0,1,2], target_names=["Low", "Medium", "High"], output_dict=True, zero_division=0)}
                classification_results[name] = metrics
                classification_models[name] = model
                joblib.dump({"model": model, "features": list(X_train.columns), "name": name, "low_cutoff": low_cut, "high_cutoff": high_cut, "classes": {0: "Low", 1: "Medium", 2: "High"}}, self.model_dir / f"classification_{name}.joblib", compress=3)
                logger.info("%s classification results: accuracy=%s macro_f1=%s", name, metrics["accuracy"], metrics["macro_f1"])
            best_clf_name = max(classification_results, key=lambda n: classification_results[n]["macro_f1"])
            # Store concise metrics and detailed model comparison separately.
            metrics_doc = {"split": "earliest 80% train, latest 20% test by datetime", "train_rows": len(train), "test_rows": len(test), "classification_cutoffs_train_only": {"low_max": low_cut, "medium_max": high_cut}, "regression": {"best_model": best_reg_name, "models": regression_results}, "classification": {"best_model": best_clf_name, "models": classification_results}}
            save_json(metrics_doc, self.model_dir / "metrics.json")
            save_json({"regression": [{"model": k, **{m:v for m,v in v0.items() if m != "best_params"}, "best_params": v0["best_params"]} for k,v0 in regression_results.items()], "classification": [{"model": k, "accuracy": v0["accuracy"], "macro_f1": v0["macro_f1"], "best_params": v0["best_params"]} for k,v0 in classification_results.items()]}, self.model_dir / "model_comparison.json")
            self._save_comparison_plots(regression_results, classification_results, classification_models[best_clf_name], X_test, yc_test)
            self._save_tree_plot(regression_models.get("decision_tree"), X_train)
            self._save_feature_importance(regression_models.get("random_forest"), X_train)
            logger.info("Best regression model: %s; best classifier: %s", best_reg_name, best_clf_name)
            return metrics_doc
        except Exception as exc:
            logger.exception("Model training failed")
            raise CustomException(f"Model training failed: {exc}") from exc

    def _save_actual_plot(self, actual, predicted):
        plt.figure(figsize=(8,5)); plt.scatter(actual, predicted, alpha=.45); plt.xlabel("Actual bike count"); plt.ylabel("Predicted bike count"); plt.title("Actual vs predicted (best regression model)"); plt.tight_layout(); plt.savefig(self.model_dir / "actual_vs_predicted.png"); plt.close()

    def _save_comparison_plots(self, reg, clf, best_model, X_test, y_test):
        plt.figure(figsize=(9,5)); plt.bar(list(reg), [v["R2"] for v in reg.values()]); plt.ylabel("R²"); plt.xticks(rotation=35, ha="right"); plt.title("Regression model comparison"); plt.tight_layout(); plt.savefig(self.model_dir / "regression_comparison.png"); plt.close()
        plt.figure(figsize=(9,5)); plt.bar(list(clf), [v["accuracy"] for v in clf.values()]); plt.ylabel("Accuracy"); plt.xticks(rotation=35, ha="right"); plt.title("Classification model comparison"); plt.tight_layout(); plt.savefig(self.model_dir / "classification_comparison.png"); plt.close()
        pred = best_model.predict(X_test); cm = confusion_matrix(y_test, pred, labels=[0,1,2]); plt.figure(figsize=(5,4)); plt.imshow(cm); plt.xticks([0,1,2],["Low","Medium","High"]); plt.yticks([0,1,2],["Low","Medium","High"]); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Best classifier confusion matrix");
        for i in range(3):
            for j in range(3): plt.text(j,i,str(cm[i,j]),ha="center",va="center")
        plt.tight_layout(); plt.savefig(self.model_dir / "best_confusion_matrix.png"); plt.close()

    def _save_tree_plot(self, model, X):
        if model is None: return
        try:
            tree = model.named_steps["model"]
            plt.figure(figsize=(18,8)); plot_tree(tree, max_depth=3, feature_names=None, filled=False, rounded=True); plt.title("Decision Tree Regressor (first 3 levels)"); plt.tight_layout(); plt.savefig(self.model_dir / "decision_tree.png"); plt.close()
        except Exception: logger.exception("Could not save decision tree plot")

    def _save_feature_importance(self, model, X):
        try:
            forest = model.named_steps["model"]
            names = model.named_steps["preprocessor"].get_feature_names_out()
            imp = pd.Series(forest.feature_importances_, index=names).sort_values().tail(18)
            plt.figure(figsize=(9,6)); imp.plot(kind="barh"); plt.title("Random Forest feature importance"); plt.tight_layout(); plt.savefig(self.model_dir / "random_forest_importance.png"); plt.close()
        except Exception: logger.exception("Could not save feature importance plot")
