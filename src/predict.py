"""
predict.py
----------
Loads persisted model/preprocessor artifacts and runs inference on new
loan applications.
"""

import joblib
import pandas as pd

from src import config
from src.exceptions import ModelNotFoundError, PredictionError
from src.logger import get_logger
from src.preprocessing import coerce_categorical_dtypes

logger = get_logger(__name__)


def load_artifacts():
    """Load the trained model and preprocessor from disk.

    Raises
    ------
    ModelNotFoundError
        If training has not been run yet.
    """
    if not config.MODEL_PATH.exists() or not config.SCALER_PATH.exists():
        raise ModelNotFoundError(
            "No trained model found. Run `python -m src.train` before predicting."
        )
    try:
        model = joblib.load(config.MODEL_PATH)
        preprocessor = joblib.load(config.SCALER_PATH)
    except Exception as exc:
        logger.exception("Failed to load model artifacts")
        raise ModelNotFoundError(f"Could not load model artifacts: {exc}") from exc

    logger.info("Model and preprocessor loaded successfully")
    return model, preprocessor


def predict_loan_eligibility(applicant_df: pd.DataFrame, model=None, preprocessor=None) -> pd.DataFrame:
    """Predict loan approval for one or more applicant records.

    Parameters
    ----------
    applicant_df : pd.DataFrame
        Must contain all columns in config.NUMERIC_FEATURES + CATEGORICAL_FEATURES.

    Returns
    -------
    pd.DataFrame with `loan_approved` (Yes/No) and `approval_probability` columns.
    """
    if model is None or preprocessor is None:
        model, preprocessor = load_artifacts()

    required = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    missing = set(required) - set(applicant_df.columns)
    if missing:
        raise PredictionError(f"Input data is missing required columns: {missing}")

    try:
        X = applicant_df[required]
        X = coerce_categorical_dtypes(X)
        X_t = preprocessor.transform(X)
        probabilities = model.predict_proba(X_t)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise PredictionError(f"Failed to generate predictions: {exc}") from exc

    result = applicant_df.copy()
    result["approval_probability"] = probabilities.round(4)
    result["loan_approved"] = pd.Series(predictions).map({1: "Yes", 0: "No"}).values
    logger.info("Generated predictions for %d record(s)", len(result))
    return result
