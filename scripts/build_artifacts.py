"""
scripts/build_artifacts.py

Run this AFTER notebooks/01_churn_pipeline.ipynb has trained the models and
saved them under models/ (e.g. models/XGBoost_model.pkl), to fit and save
the preprocessor the Streamlit app needs at inference time.

Usage:
    python scripts/build_artifacts.py

Outputs:
    models/preprocessor.pkl
"""

import os
import sys

import joblib
import pandas as pd

# Allow running this script directly from the repo root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import ChurnPreprocessor  # noqa: E402

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'BankChurners.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
ARTIFACT_PATH = os.path.join(MODELS_DIR, 'preprocessor.pkl')

# Matches the model dict key used in notebooks/01_churn_pipeline.ipynb
# ("XGBoost": XGBClassifier()) — the notebook saves it as models/XGBoost_model.pkl
CHAMPION_MODEL_NAME = "XGBoost"
CHAMPION_MODEL_PATH = os.path.join(MODELS_DIR, f"{CHAMPION_MODEL_NAME}_model.pkl")


def load_champion_feature_columns() -> list[str] | None:
    """
    The notebook only saves the raw model pickle, not a separate
    feature_columns.pkl. If the champion model exposes feature_names_in_
    (set automatically when a sklearn-compatible estimator is fit on a
    DataFrame), use that as the serving contract so inference matches
    training exactly. Otherwise the preprocessor keeps all columns it
    produces from the data.
    """
    if not os.path.exists(CHAMPION_MODEL_PATH):
        print(f"No champion model found at: {CHAMPION_MODEL_PATH}")
        print("Run notebooks/01_churn_pipeline.ipynb first to train and save it.")
        return None

    model = joblib.load(CHAMPION_MODEL_PATH)
    feature_columns = getattr(model, "feature_names_in_", None)
    if feature_columns is None:
        print(
            f"{CHAMPION_MODEL_NAME} model has no feature_names_in_. "
            "The preprocessor will keep all transformed feature columns."
        )
        return None

    feature_columns = list(feature_columns)
    print(f"Loaded {len(feature_columns)} feature columns from: {CHAMPION_MODEL_PATH}")
    return feature_columns


def main():
    print(f"Loading raw data from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    print("Raw shape:", df.shape)

    preprocessor = ChurnPreprocessor(feature_columns=load_champion_feature_columns()).fit(df)

    preprocessor.save(ARTIFACT_PATH)
    print(f"Saved preprocessor to: {ARTIFACT_PATH}")
    print(f"Final feature columns ({len(preprocessor.final_feature_columns)}):")
    for c in preprocessor.final_feature_columns:
        print(" -", c)


if __name__ == "__main__":
    main()