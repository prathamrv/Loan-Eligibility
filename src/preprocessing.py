"""
preprocessing.py
-----------------
Builds a scikit-learn ColumnTransformer that scales numeric features and
one-hot-encodes categorical features, mirroring the get_dummies + MinMaxScaler
approach from the original notebook but persisted as a single fitted object
so training-time and inference-time transforms are always identical.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src import config
from src.exceptions import DataValidationError
from src.logger import get_logger

logger = get_logger(__name__)


def build_preprocessor() -> ColumnTransformer:
    """Construct (but do not fit) the preprocessing ColumnTransformer."""
    logger.info(
        "Building preprocessor: %d numeric, %d categorical features",
        len(config.NUMERIC_FEATURES), len(config.CATEGORICAL_FEATURES),
    )
    return ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAL_FEATURES),
        ]
    )


def get_output_feature_names(preprocessor: ColumnTransformer) -> list:
    """Return human-readable feature names after transformation."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception as exc:  # pragma: no cover
        logger.warning("Falling back to generic feature names: %s", exc)
        return []


def coerce_categorical_dtypes(X: pd.DataFrame) -> pd.DataFrame:
    """Force categorical feature columns to plain Python `object` dtype.

    Pandas 3.x defaults string columns to its new StringDtype backend, which
    is incompatible with scikit-learn's OneHotEncoder unknown-category check
    when a ColumnTransformer mixes those columns with numeric-but-object
    columns (like Credit_History cast to object). Casting explicitly to
    `object` avoids a `TypeError: ufunc 'isnan' not supported` at both fit
    and inference time -- this must be applied identically in both places.
    """
    X = X.copy()
    X[config.CATEGORICAL_FEATURES] = X[config.CATEGORICAL_FEATURES].astype(object)
    return X


def split_features_target(df: pd.DataFrame):
    """Split a cleaned dataframe into X (features) and y (target)."""
    required = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    missing = set(required) - set(df.columns)
    if missing:
        raise DataValidationError(f"Cannot split features/target, missing columns: {missing}")

    X = df[required]
    y = df[config.TARGET_COLUMN] if config.TARGET_COLUMN in df.columns else None
    return X, y
