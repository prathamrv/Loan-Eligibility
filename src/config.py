"""
config.py
---------
Centralized configuration for the Loan Eligibility Prediction project.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

RAW_DATA_PATH = DATA_DIR / "credit.csv"
MODEL_PATH = MODEL_DIR / "loan_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"

for _dir in (DATA_DIR, MODEL_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

ID_COLUMN = "Loan_ID"
TARGET_COLUMN = "Loan_Approved"

NUMERIC_FEATURES = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]

# Loan_Amount_Term and Credit_History are cast to 'object' in the original
# notebook because they're categorical/ordinal in nature despite being
# numeric-looking, so they're one-hot encoded alongside the true
# categorical columns.
CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

RF_PARAMS = {
    "n_estimators": 200,
    "random_state": RANDOM_STATE,
}

LOG_FILE = LOG_DIR / "loan_pipeline.log"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
