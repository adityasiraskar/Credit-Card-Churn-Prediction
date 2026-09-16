from pathlib import Path
import sys

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import ChurnPreprocessor  # noqa: E402

RAW_DATA_PATH = ROOT / "data" / "raw" / "BankChurners.csv"

# notebooks/01_churn_pipeline.ipynb saves each model as "{name}_model.pkl"
# directly under models/ (no saved_models/ subfolder, no best_model_ prefix,
# no separate feature_columns.pkl file).
MODEL_PATH = ROOT / "models" / "XGBoost_model.pkl"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.pkl"


def _model_input(model, features_df):
    if getattr(model, "feature_names_in_", None) is not None:
        return features_df
    return features_df.to_numpy()


def test_preprocessor_matches_champion_model_features():
    model = joblib.load(MODEL_PATH)
    raw_df = pd.read_csv(RAW_DATA_PATH)

    # The notebook never saves a feature_columns.pkl, so the champion
    # model's own feature_names_in_ (set by sklearn when it's fit on a
    # DataFrame) is the serving contract instead.
    expected_columns = list(getattr(model, "feature_names_in_", []))

    preprocessor = ChurnPreprocessor(feature_columns=expected_columns or None).fit(raw_df)
    features_df = preprocessor.transform(
        raw_df.head(5).drop(columns=["CLIENTNUM", "Attrition_Flag"], errors="ignore")
    )

    if expected_columns:
        assert list(features_df.columns) == expected_columns


def test_saved_model_accepts_saved_preprocessor_output():
    model = joblib.load(MODEL_PATH)
    preprocessor = ChurnPreprocessor.load(PREPROCESSOR_PATH)
    raw_df = pd.read_csv(RAW_DATA_PATH).head(3)
    features_df = preprocessor.transform(
        raw_df.drop(columns=["CLIENTNUM", "Attrition_Flag"], errors="ignore")
    )

    assert features_df.shape[1] == getattr(model, "n_features_in_", features_df.shape[1])

    probabilities = model.predict_proba(_model_input(model, features_df))
    assert probabilities.shape == (3, 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()