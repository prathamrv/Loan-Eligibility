"""
app.py
------
Streamlit front-end for the Loan Eligibility Prediction project.

Two pages:
  1. Predict Eligibility   - assess a single loan application
  2. Model Insights         - model comparison metrics

Run with: streamlit run app.py
"""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.data_loader import clean_data, load_raw_data
from src.exceptions import LoanPipelineError, ModelNotFoundError
from src.logger import get_logger
from src.predict import load_artifacts, predict_loan_eligibility

logger = get_logger(__name__)

st.set_page_config(page_title="Loan Eligibility Predictor", page_icon="🏦", layout="wide")


@st.cache_resource(show_spinner="Loading trained model...")
def get_model_and_preprocessor():
    return load_artifacts()


st.sidebar.title("🏦 Loan Eligibility")
page = st.sidebar.radio("Navigate", ["Predict Eligibility", "Model Insights"])
st.sidebar.markdown("---")
st.sidebar.caption(
    "CST2216 — Individual Term Project\n\n"
    "Loan Eligibility Prediction · Algonquin College"
)

try:
    model, preprocessor = get_model_and_preprocessor()
except ModelNotFoundError:
    st.sidebar.error("No trained model found.")
    st.error(
        "⚠️ No trained model artifacts were found under `/models`.\n\n"
        "Run `python -m src.train` from the project root first, then reload this app."
    )
    st.stop()


if page == "Predict Eligibility":
    st.title("Predict Loan Eligibility")
    st.write("Enter an applicant's details to estimate their likelihood of loan approval.")

    with st.form("applicant_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])

        with col2:
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            applicant_income = st.number_input("Applicant Monthly Income ($)", 0, 100000, 5000)
            coapplicant_income = st.number_input("Co-applicant Monthly Income ($)", 0, 50000, 0)
            loan_amount = st.number_input("Loan Amount (in $1,000s)", 0, 1000, 150)

        with col3:
            loan_term = st.selectbox("Loan Term (months)", [360, 180, 120, 60, 300, 240, 84, 36, 12], index=0)
            credit_history = st.selectbox("Has Good Credit History", [1, 0], format_func=lambda x: "Yes" if x else "No")
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        submitted = st.form_submit_button("Check Eligibility", type="primary")

    if submitted:
        applicant = pd.DataFrame([{
            "Gender": gender, "Married": married, "Dependents": dependents,
            "Education": education, "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history, "Property_Area": property_area,
        }])

        try:
            result = predict_loan_eligibility(applicant, model=model, preprocessor=preprocessor)
            approved = result["loan_approved"].iloc[0]
            prob = float(result["approval_probability"].iloc[0])

            st.markdown("---")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Decision", "✅ Approved" if approved == "Yes" else "❌ Not Approved")
            with c2:
                st.metric("Approval Probability", f"{prob:.1%}")
                st.progress(prob)

        except LoanPipelineError as exc:
            logger.error("Prediction error in app: %s", exc)
            st.error(f"Could not generate a prediction: {exc}")

elif page == "Model Insights":
    st.title("Model Insights")

    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH) as f:
            metrics = json.load(f)

        st.subheader("Model Comparison (Held-out Test Set)")
        cols = st.columns(len(metrics["models"]))
        for col, (name, m) in zip(cols, metrics["models"].items()):
            with col:
                is_selected = name == metrics["selected_model"]
                st.markdown(f"**{name.replace('_', ' ').title()}**" + (" ✅" if is_selected else ""))
                st.metric("Accuracy", f"{m['accuracy']:.1%}")
                st.metric("F1 Score", f"{m['f1_score']:.3f}")

        st.caption(
            f"Selected model: **{metrics['selected_model'].replace('_', ' ').title()}** "
            f"(highest accuracy on {metrics['n_test']} held-out applications). "
            f"Target success criteria was 76% accuracy."
        )
    else:
        st.warning("No metrics file found. Run training first.")

    st.markdown("---")
    st.subheader("Dataset Overview")
    raw_df = clean_data(load_raw_data())
    d1, d2 = st.columns(2)
    with d1:
        approval_counts = raw_df[config.TARGET_COLUMN].map({1: "Approved", 0: "Denied"}).value_counts()
        st.plotly_chart(px.pie(values=approval_counts.values, names=approval_counts.index,
                                title="Loan Approval Distribution"), use_container_width=True)
    with d2:
        st.plotly_chart(px.histogram(raw_df, x="ApplicantIncome",
                                      color=raw_df[config.TARGET_COLUMN].map({1: "Approved", 0: "Denied"}),
                                      title="Applicant Income by Approval Status", barmode="overlay", opacity=0.7),
                         use_container_width=True)
