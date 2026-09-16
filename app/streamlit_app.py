"""
app/streamlit_app.py

Interactive Streamlit demo for the Customer Churn Prediction project.
Uses the same preprocessor (src/preprocessing.py) and trained model to
produce churn predictions from raw customer inputs.

Run locally:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import glob

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import ChurnPreprocessor  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'preprocessor.pkl')

# Champion model from notebooks/01_churn_pipeline.ipynb and
# notebooks/02_churn_pipeline_using_smote.ipynb: XGBoost trained without
# SMOTE gave the best precision/F1 trade-off (95% churn precision, 91%
# churn F1-score, 97% accuracy, 14 false positives). The notebook saves
# each model as "{name}_model.pkl" directly under models/ — matching the
# "XGBoost" key in its models dict — so the file is models/XGBoost_model.pkl.
CHAMPION_MODEL_NAME = "XGBoost"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="wide",
)


# ----------------------------------------------------------------------
# Cached resource loading
# ----------------------------------------------------------------------
@st.cache_resource
def load_model_and_preprocessor():
    champion_path = os.path.join(MODELS_DIR, f"{CHAMPION_MODEL_NAME}_model.pkl")

    if os.path.exists(champion_path):
        model_path = champion_path
    else:
        model_matches = glob.glob(os.path.join(MODELS_DIR, "*_model.pkl"))
        if not model_matches:
            return None, None, None
        model_path = max(model_matches, key=os.path.getmtime)

    model = joblib.load(model_path)
    model_name = os.path.basename(model_path).replace("_model.pkl", "")

    if not os.path.exists(PREPROCESSOR_PATH):
        return model, model_name, None

    preprocessor = ChurnPreprocessor.load(PREPROCESSOR_PATH)
    return model, model_name, preprocessor


model, model_name, preprocessor = load_model_and_preprocessor()


def artifact_mismatch_message() -> str | None:
    if model is None or preprocessor is None:
        return None

    expected = getattr(model, "n_features_in_", None)
    actual = len(getattr(preprocessor, "final_feature_columns", []))
    if expected is not None and actual != expected:
        return (
            "Model/preprocessor feature mismatch: the model expects "
            f"{expected} features, but the preprocessor produces {actual}. "
            "Run `python scripts/build_artifacts.py` after training the model."
        )

    return None


def predict_churn_probabilities(features_df: pd.DataFrame) -> np.ndarray:
    model_input = (
        features_df
        if getattr(model, "feature_names_in_", None) is not None
        else features_df.to_numpy()
    )
    probabilities = model.predict_proba(model_input)

    classes = list(getattr(model, "classes_", []))
    churn_index = classes.index(1) if 1 in classes else 1
    return probabilities[:, churn_index]


def risk_level(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    elif probability < 0.65:
        return "Medium"
    return "High"


def risk_color(level: str) -> str:
    return {"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"}[level]


def predict_one(customer: dict) -> dict:
    raw_df = pd.DataFrame([customer])
    features_df = preprocessor.transform(raw_df)
    proba = float(predict_churn_probabilities(features_df)[0])
    return {
        "prediction": "Churn" if proba >= 0.5 else "Stay",
        "probability": proba,
        "risk": risk_level(proba),
    }


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🏦 Customer Churn Predictor")
st.caption(
    "Based on *Customer Churn Prediction in the Banking Sector Using Machine "
    "Learning-Based Classification Models* (Tran, Le & Nguyen, 2023, IJIKM Vol. 18)"
)

if model is None:
    st.error(
        "⚠️ No trained model found in `models/`. "
        "Run `notebooks/01_churn_pipeline.ipynb` first to train and save a model."
    )
    st.stop()

if preprocessor is None:
    st.error(
        "⚠️ No preprocessor found. Run `python scripts/build_artifacts.py` first."
    )
    st.stop()

mismatch_message = artifact_mismatch_message()
if mismatch_message:
    st.error(mismatch_message)
    st.stop()

st.success(f"✅ Model loaded: **{model_name}**")

tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📂 Batch Prediction (CSV)", "ℹ️ About"])

# ----------------------------------------------------------------------
# TAB 1 — Single prediction form
# ----------------------------------------------------------------------
with tab1:
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        customer_age = st.slider("Customer Age", 18, 80, 45)
        gender = st.selectbox("Gender", ["M", "F"])
        dependent_count = st.slider("Dependent Count", 0, 10, 3)
        education_level = st.selectbox(
            "Education Level",
            ["Uneducated", "High School", "Graduate", "Post-Graduate", "Doctorate"],
            index=2,
        )
        marital_status = st.selectbox("Marital Status", ["Married", "Single"])
        income_category = st.selectbox(
            "Income Category",
            ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"],
            index=2,
        )
        card_category = st.selectbox("Card Category", ["Blue", "Silver", "Gold", "Platinum"])

    with col2:
        st.markdown("**Relationship with Bank**")
        months_on_book = st.slider("Months on Book", 6, 60, 36)
        total_relationship_count = st.slider("Total Relationship Count (products held)", 1, 6, 5)
        months_inactive = st.slider("Months Inactive (last 12 mo)", 0, 12, 1)
        contacts_count = st.slider("Contacts Count (last 12 mo)", 0, 10, 2)

    with col3:
        st.markdown("**Transaction Behavior**")
        credit_limit = st.number_input("Credit Limit ($)", min_value=0.0, value=8500.0, step=100.0)
        total_revolving_bal = st.number_input("Total Revolving Balance ($)", min_value=0.0, value=1200.0, step=50.0)
        avg_open_to_buy = st.number_input("Avg Open To Buy ($)", min_value=0.0, value=7300.0, step=100.0)
        total_amt_chng = st.slider("Total Amt Change Q4/Q1", 0.0, 3.0, 0.76, 0.01)
        total_trans_amt = st.number_input("Total Transaction Amount ($, last 12mo)", min_value=0.0, value=4500.0, step=100.0)
        total_trans_ct = st.slider("Total Transaction Count (last 12mo)", 0, 150, 65)
        total_ct_chng = st.slider("Total Count Change Q4/Q1", 0.0, 3.0, 0.70, 0.01)
        avg_utilization = st.slider("Avg Utilization Ratio", 0.0, 1.0, 0.14, 0.01)

    st.divider()

    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        customer = {
            "Customer_Age": customer_age,
            "Gender": gender,
            "Dependent_count": dependent_count,
            "Education_Level": education_level,
            "Marital_Status": marital_status,
            "Income_Category": income_category,
            "Card_Category": card_category,
            "Months_on_book": months_on_book,
            "Total_Relationship_Count": total_relationship_count,
            "Months_Inactive_12_mon": months_inactive,
            "Contacts_Count_12_mon": contacts_count,
            "Credit_Limit": credit_limit,
            "Total_Revolving_Bal": total_revolving_bal,
            "Avg_Open_To_Buy": avg_open_to_buy,
            "Total_Amt_Chng_Q4_Q1": total_amt_chng,
            "Total_Trans_Amt": total_trans_amt,
            "Total_Trans_Ct": total_trans_ct,
            "Total_Ct_Chng_Q4_Q1": total_ct_chng,
            "Avg_Utilization_Ratio": avg_utilization,
        }

        result = predict_one(customer)

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric("Prediction", result["prediction"])

        with res_col2:
            st.metric("Churn Probability", f"{result['probability'] * 100:.1f}%")

        with res_col3:
            color = risk_color(result["risk"])
            st.markdown(
                f"<div style='padding:10px;border-radius:8px;background-color:{color}22;"
                f"border:2px solid {color};text-align:center;'>"
                f"<span style='color:{color};font-weight:bold;font-size:1.1em;'>"
                f"Risk Level: {result['risk']}</span></div>",
                unsafe_allow_html=True,
            )

        # Probability gauge-style bar
        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.barh([0], [1], color="#e0e0e0")
        ax.barh([0], [result["probability"]], color=risk_color(result["risk"]))
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.3, 0.65, 1.0])
        ax.set_xticklabels(['0%', '30% (Low/Med)', '65% (Med/High)', '100%'])
        ax.set_title(f"Churn Probability: {result['probability']*100:.1f}%")
        st.pyplot(fig)

        if result["prediction"] == "Churn":
            st.warning(
                "⚠️ This customer is **likely to churn**. Consider proactive retention "
                "actions: personalized offers, relationship manager outreach, or fee waivers."
            )
        else:
            st.info("✅ This customer is **likely to stay**. No immediate action needed.")

# ----------------------------------------------------------------------
# TAB 2 — Batch prediction via CSV upload
# ----------------------------------------------------------------------
with tab2:
    st.subheader("Upload a CSV of Customers")
    st.markdown(
        "CSV must contain these raw columns (same names as the Kaggle dataset, "
        "without `CLIENTNUM` or `Attrition_Flag`):"
    )
    st.code(
        "Customer_Age, Gender, Dependent_count, Education_Level, Marital_Status, "
        "Income_Category, Card_Category, Months_on_book, Total_Relationship_Count, "
        "Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit, "
        "Total_Revolving_Bal, Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, "
        "Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio",
        language="text",
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(batch_df.head())

        if st.button("🔮 Run Batch Prediction", type="primary"):
            try:
                features_df = preprocessor.transform(batch_df)
                probas = predict_churn_probabilities(features_df)

                results_df = batch_df.copy()
                results_df["Churn_Probability"] = np.round(probas, 4)
                results_df["Prediction"] = np.where(probas >= 0.5, "Churn", "Stay")
                results_df["Risk_Level"] = [risk_level(p) for p in probas]

                st.success(f"Predicted churn for {len(results_df)} customers.")
                st.dataframe(results_df)

                churn_count = (results_df["Prediction"] == "Churn").sum()
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Predicted to Churn", f"{churn_count} / {len(results_df)}")
                with col_b:
                    st.metric("Churn Rate", f"{churn_count / len(results_df) * 100:.1f}%")

                fig, ax = plt.subplots(figsize=(6, 4))
                results_df["Risk_Level"].value_counts().reindex(
                    ["Low", "Medium", "High"]
                ).plot(kind="bar", color=["#2E7D32", "#F9A825", "#C62828"], ax=ax)
                ax.set_title("Risk Level Distribution")
                ax.set_ylabel("Number of Customers")
                st.pyplot(fig)

                csv_out = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Predictions as CSV",
                    data=csv_out,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Make sure your CSV has exactly the expected raw column names.")

# ----------------------------------------------------------------------
# TAB 3 — About
# ----------------------------------------------------------------------
with tab3:
    st.subheader("About This Project")
    st.markdown(
        """
This demo is part of an end-to-end **Customer Churn Prediction** project for the
banking sector, inspired by:

> Tran, H., Le, N., & Nguyen, V.-H. (2023). *Customer churn prediction in the banking
> sector using machine learning-based classification models.* Interdisciplinary Journal
> of Information, Knowledge, and Management, 18, 87-105.

**Pipeline:**
1. **EDA** — explored distributions, target imbalance, correlations
2. **Preprocessing** — cleaned unknowns, encoded categoricals, scaled numeric features
3. **Customer Segmentation** — K-Means clustering (k=6, chosen via the elbow method)
4. **Modeling** — 8 classifiers (Logistic Regression, Naive Bayes, Decision Tree,
   Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM), evaluated with
   and without SMOTE oversampling, all runs tracked in MLflow
   (`notebooks/01_churn_pipeline.ipynb` and `notebooks/02_churn_pipeline_using_smote.ipynb`)
5. **Deployment** — this Streamlit app, using the exact same preprocessing logic as
   training (`src/preprocessing.py`)

**Key finding:** across all eight classifiers, **XGBoost without SMOTE** gave the
best overall trade-off — 95% churn precision, 91% churn F1-score, 97% accuracy, and
only 14 false positives. LightGBM with SMOTE reached higher churn recall (91%) but
with more false positives, making it a reasonable alternative when missing a
churner is costlier than a false alarm.

**Tech stack:** Python, pandas, scikit-learn, imbalanced-learn, and Streamlit.
        """
    )
    st.caption(f"Currently serving model: **{model_name}**")