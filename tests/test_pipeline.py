"""
test_pipeline.py
-----------------
Unit tests for the loan eligibility pipeline. Run with: pytest tests/ -v
"""

import pandas as pd
import pytest

from src import config
from src.data_loader import clean_data, load_raw_data
from src.exceptions import DataLoadError, DataValidationError, PredictionError
from src.preprocessing import build_preprocessor, split_features_target
from src.predict import predict_loan_eligibility


def test_load_raw_data_success():
    df = load_raw_data()
    assert not df.empty
    assert config.TARGET_COLUMN in df.columns


def test_load_raw_data_missing_file():
    with pytest.raises(DataLoadError):
        load_raw_data(path="data/does_not_exist.csv")


def test_clean_data_no_missing_values():
    df = clean_data(load_raw_data())
    required = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES + [config.TARGET_COLUMN]
    assert df[required].isna().sum().sum() == 0


def test_clean_data_target_is_binary():
    df = clean_data(load_raw_data())
    assert set(df[config.TARGET_COLUMN].unique()) <= {0, 1}


def test_split_features_target_shapes():
    df = clean_data(load_raw_data())
    X, y = split_features_target(df)
    assert len(X) == len(y)
    assert config.TARGET_COLUMN not in X.columns


def test_preprocessor_transforms_without_error():
    df = clean_data(load_raw_data())
    X, _ = split_features_target(df)
    preprocessor = build_preprocessor()
    X_t = preprocessor.fit_transform(X)
    assert X_t.shape[0] == len(X)
    assert X_t.shape[1] > len(config.NUMERIC_FEATURES)


def test_predict_missing_columns_raises():
    if not config.MODEL_PATH.exists():
        pytest.skip("Model not trained yet; run `python -m src.train` first.")
    bad_df = pd.DataFrame({"ApplicantIncome": [5000]})
    with pytest.raises(PredictionError):
        predict_loan_eligibility(bad_df)


def test_predict_returns_expected_columns():
    if not config.MODEL_PATH.exists():
        pytest.skip("Model not trained yet; run `python -m src.train` first.")
    df = clean_data(load_raw_data()).drop(columns=[config.TARGET_COLUMN]).head(5)
    result = predict_loan_eligibility(df)
    assert "loan_approved" in result.columns
    assert "approval_probability" in result.columns
    assert result["approval_probability"].between(0, 1).all()
