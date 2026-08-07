# Loan Eligibility Prediction

**Course:** CST2216 — Individual Term Project (Modularizing and Deploying ML Code)
**Author:** Pratham Varde
**Source:** Modularized from `Loan_Eligibility_Model_Solution.ipynb` (Level 1, Week 10)

A classification pipeline that predicts whether a bank loan application
should be approved, comparing Logistic Regression, Decision Tree, and
Random Forest classifiers, deployed as an interactive Streamlit app.

## Project Structure

```
loan_eligibility/
├── app.py                  # Streamlit application (Predict Eligibility / Model Insights)
├── main.py                 # CLI entry point: runs the full training pipeline
├── requirements.txt
├── README.md
├── data/credit.csv          # 614 loan applications (13 attributes)
├── models/                  # Generated at train time
├── logs/                    # Rotating log file
├── src/
│   ├── config.py             # Paths, feature list, hyperparameters
│   ├── logger.py              # Centralized logging setup
│   ├── exceptions.py          # Custom exception hierarchy
│   ├── data_loader.py          # Load + impute missing values
│   ├── preprocessing.py        # ColumnTransformer (MinMaxScaler + OneHotEncoder)
│   ├── train.py                # Train/compare/persist 3 classifiers
│   └── predict.py              # Load artifacts + run inference
└── tests/
    └── test_pipeline.py         # Pytest unit tests
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model

```bash
python main.py
```

Trains Logistic Regression, Decision Tree, and Random Forest, evaluates
each on a held-out test set, and persists whichever has the highest
accuracy to `models/loan_model.joblib`.

## Run the app

```bash
streamlit run app.py
```

## Run tests

```bash
pytest tests/ -v
```

## Model Performance (held-out test set, 123 applications)

| Model | Accuracy | F1 |
|---|---|---|
| **Logistic Regression (selected)** | **84.6%** | **0.897** |
| Random Forest | 82.1% | 0.876 |
| Decision Tree | 77.2% | 0.835 |

Success criterion from the original notebook was ≥76% accuracy — all
three models clear this bar.

## Design Notes

- **Missing-value imputation** mirrors the original notebook exactly:
  categorical variables (Gender, Married, Dependents, Self_Employed,
  Loan_Amount_Term, Credit_History) are imputed with the column mode;
  the numeric `LoanAmount` is imputed with the median.
- `Credit_History` and `Loan_Amount_Term` are cast to `object` dtype and
  one-hot encoded rather than scaled, matching the original notebook's
  treatment of them as ordinal/categorical rather than continuous.
- **Real bug found and fixed during modularization:** pandas 3.x's new
  default string dtype is incompatible with scikit-learn's
  `OneHotEncoder` unknown-category check when mixed with numeric-as-object
  columns in the same `ColumnTransformer` — this raised a cryptic
  `TypeError: ufunc 'isnan' not supported` at inference time. Fixed by
  explicitly coercing categorical columns to plain Python `object` dtype
  identically at both fit and inference time (see
  `preprocessing.coerce_categorical_dtypes`).
- The fitted preprocessor is persisted alongside the model so training-time
  and inference-time transforms are always identical.

## Limitations

- Dataset is a single historical snapshot of 614 applications; a
  production model would need periodic retraining and monitoring for
  demographic/economic drift.
- Logistic Regression was selected purely on accuracy; a real deployment
  should also weigh the cost of false approvals vs. false denials, which
  may favor a different operating threshold or model.
