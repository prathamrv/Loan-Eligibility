"""
data_loader.py
--------------
Loads and validates the loan eligibility dataset, then applies the
missing-value imputation strategy from the original notebook:
  - Categorical variables: mode imputation
  - Numerical variables (LoanAmount): median imputation
"""

import pandas as pd

from src import config
from src.exceptions import DataLoadError, DataValidationError
from src.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = (
    [config.ID_COLUMN, config.TARGET_COLUMN] + config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
)


def load_raw_data(path=None) -> pd.DataFrame:
    """Load the raw loan eligibility CSV from disk and run basic sanity checks.

    Raises
    ------
    DataLoadError
        If the file cannot be found or parsed.
    DataValidationError
        If required columns are missing or the file is empty.
    """
    path = path or config.RAW_DATA_PATH
    logger.info("Loading raw data from %s", path)

    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        logger.error("Data file not found at %s", path)
        raise DataLoadError(f"Could not find data file at {path}") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"Could not parse CSV at {path}: {exc}") from exc

    if df.empty:
        raise DataValidationError("Loaded dataset contains zero rows.")

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"Dataset is missing required columns: {missing_cols}")

    logger.info("Loaded dataframe with shape %s", df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and normalize the target column, mirroring the
    cleaning strategy from the original notebook."""
    logger.info("Cleaning data (shape before: %s)", df.shape)
    df = df.copy()

    # Categorical variables: impute with mode
    for col in ["Gender", "Married", "Dependents", "Self_Employed", "Loan_Amount_Term", "Credit_History"]:
        n_missing = df[col].isna().sum()
        if n_missing:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.warning("Imputed %d missing '%s' values with mode: %s", n_missing, col, mode_val)

    # Numerical variable: impute with median
    n_missing = df["LoanAmount"].isna().sum()
    if n_missing:
        median_val = df["LoanAmount"].median()
        df["LoanAmount"] = df["LoanAmount"].fillna(median_val)
        logger.warning("Imputed %d missing LoanAmount values with median: %.1f", n_missing, median_val)

    # Cast ordinal-but-numeric-looking columns to object, as in the original notebook
    df["Credit_History"] = df["Credit_History"].astype("object")
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype("object")

    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].map({"Y": 1, "N": 0})
    if df[config.TARGET_COLUMN].isna().any():
        raise DataValidationError("Target column contains values other than Y/N.")

    remaining_nulls = df[REQUIRED_COLUMNS].isna().sum().sum()
    if remaining_nulls:
        logger.warning("%d missing values remain after imputation; dropping affected rows", remaining_nulls)
        df = df.dropna(subset=REQUIRED_COLUMNS)

    logger.info("Cleaning complete (shape after: %s)", df.shape)
    return df
