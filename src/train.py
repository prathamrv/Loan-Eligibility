"""
train.py
--------
Trains and compares three classifiers (Logistic Regression, Decision Tree,
Random Forest) on the loan eligibility dataset, mirroring the model
comparison in the original notebook, then persists the best-performing
model (by test accuracy, target >= 76% per the project brief).

Running `python -m src.train` regenerates every artifact under /models.
"""

import json

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.data_loader import clean_data, load_raw_data
from src.exceptions import ModelTrainingError
from src.logger import get_logger
from src.preprocessing import build_preprocessor, coerce_categorical_dtypes, split_features_target

logger = get_logger(__name__)


def _evaluate(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    logger.info("%s test metrics: %s", name, {k: v for k, v in metrics.items() if k != "confusion_matrix"})
    return metrics


def train_model(save_artifacts: bool = True) -> dict:
    """Train and compare three classifiers, then persist the most accurate one.

    Returns
    -------
    dict with per-model metrics and the selected model name.
    """
    try:
        raw_df = load_raw_data()
        clean_df = clean_data(raw_df)
        X, y = split_features_target(clean_df)
        X = coerce_categorical_dtypes(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y,
        )
        logger.info("Train/test split: %d / %d rows", len(X_train), len(X_test))

        preprocessor = build_preprocessor()
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        models = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "decision_tree": DecisionTreeClassifier(random_state=config.RANDOM_STATE),
            "random_forest": RandomForestClassifier(**config.RF_PARAMS),
        }

        results = {}
        fitted_models = {}
        for name, clf in models.items():
            clf.fit(X_train_t, y_train)
            fitted_models[name] = clf
            results[name] = _evaluate(clf, X_test_t, y_test, name)

        selected_name = max(results, key=lambda k: results[k]["accuracy"])
        selected_model = fitted_models[selected_name]
        logger.info("Selected model: %s (highest test accuracy)", selected_name)

        summary = {
            "models": results,
            "selected_model": selected_name,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    except Exception as exc:
        logger.exception("Model training failed")
        raise ModelTrainingError(f"Training pipeline failed: {exc}") from exc

    if save_artifacts:
        _persist_artifacts(selected_model, preprocessor, selected_name, summary)

    return summary


def _persist_artifacts(model, preprocessor, model_name: str, metrics: dict) -> None:
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(preprocessor, config.SCALER_PATH)

    from src.preprocessing import get_output_feature_names
    with open(config.FEATURE_NAMES_PATH, "w") as f:
        json.dump(get_output_feature_names(preprocessor), f, indent=2)

    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Artifacts saved to %s (model: %s)", config.MODEL_DIR, model_name)


if __name__ == "__main__":
    results = train_model()
    print(json.dumps(results, indent=2))
