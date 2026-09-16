"""
preprocessing.py

Single source of truth for cleaning / transforming the raw BankChurners data.
Used by:
  - notebooks/01_churn_pipeline.ipynb (training time, no-SMOTE pipeline)
  - notebooks/02_churn_pipeline_using_smote.ipynb (training time, SMOTE pipeline)
  - scripts/build_artifacts.py (fits & saves the encoders)
  - app/streamlit_app.py (inference time, on raw customer input)

Keeping this logic in one place guarantees the Streamlit app preprocesses
new customers exactly the same way the model was trained.

This mirrors exactly what the training notebooks do to the raw data:
  1. Drop CLIENTNUM and Kaggle's auto-generated Naive-Bayes helper columns.
  2. Map Attrition_Flag -> 1 (Attrited Customer) / 0 (Existing Customer).
  3. Label-encode the remaining categorical (object-dtype) columns.
  4. Leave every numeric column untouched. The notebooks do not scale or
     normalize any feature, so no StandardScaler/MinMaxScaler is used here.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------------------
# Column groups (must match notebooks/01_churn_pipeline.ipynb)
# ----------------------------------------------------------------------
ID_COLS = ['CLIENTNUM']
TARGET_COL = 'Attrition_Flag'
TARGET_MAPPING = {'Attrited Customer': 1, 'Existing Customer': 0}

# Every remaining object-dtype column after the target mapping is
# label-encoded in the notebooks (Gender, Education_Level, Marital_Status,
# Income_Category, Card_Category).
CATEGORICAL_COLS = [
    'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'
]


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop CLIENTNUM and Kaggle's auto-generated Naive-Bayes helper columns."""
    nb_cols = [c for c in df.columns if 'Naive_Bayes' in c]
    cols_to_drop = [c for c in ID_COLS + nb_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map Attrition_Flag -> 1 (Attrited Customer) / 0 (Existing Customer)."""
    df = df.copy()
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].replace(TARGET_MAPPING)
    return df


class ChurnPreprocessor:
    """
    Fits on raw training data, can be saved/loaded with joblib, and transforms
    either a full DataFrame (training) or a single-row DataFrame (inference)
    into the exact feature matrix the model expects.
    """

    def __init__(self, feature_columns: Sequence[str] | None = None):
        # One LabelEncoder per categorical column, fit at training time
        self.label_encoders: dict[str, LabelEncoder] = {}
        # category -> code lookup derived from each LabelEncoder's classes_,
        # used at transform time so an unseen category at inference doesn't
        # raise (LabelEncoder.transform would raise a ValueError instead).
        self.label_maps: dict[str, dict[str, int]] = {}
        self.final_feature_columns: list[str] = list(feature_columns or [])

    # ------------------------------------------------------------------
    # Fit (training time only)
    # ------------------------------------------------------------------
    def fit(self, raw_df: pd.DataFrame) -> "ChurnPreprocessor":
        df = drop_unused_columns(raw_df)
        df = encode_target(df)

        categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

        for col in categorical_cols:
            encoder = LabelEncoder()
            encoder.fit(df[col])
            self.label_encoders[col] = encoder
            self.label_maps[col] = {
                category: int(code) for code, category in enumerate(encoder.classes_)
            }

        # Build the transformed dataframe once to lock in column order.
        # When a trained model already saved its input columns, use those as
        # the serving contract so inference matches the model exactly.
        transformed = self._transform_core(df)
        if self.final_feature_columns:
            missing_cols = [
                c for c in self.final_feature_columns
                if c not in transformed.columns and c != TARGET_COL
            ]
            if missing_cols:
                raise ValueError(
                    "Feature columns are not produced by the preprocessor: "
                    + ", ".join(missing_cols)
                )
        else:
            self.final_feature_columns = [c for c in transformed.columns if c != TARGET_COL]

        return self

    # ------------------------------------------------------------------
    # Internal shared transform logic
    # ------------------------------------------------------------------
    def _transform_core(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, mapping in self.label_maps.items():
            if col in df.columns:
                # Unseen categories at inference fall back to -1 instead of raising
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
        return df

    # ------------------------------------------------------------------
    # Transform (training OR inference)
    # ------------------------------------------------------------------
    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw input (single row or full dataframe) into the exact
        feature matrix the model expects (same columns, same order).
        Missing columns are filled with 0; target column is dropped if present.
        """
        df = raw_df.copy()

        # Only drop ID/helper columns when they're actually present
        # (at inference time there is typically no CLIENTNUM column)
        if any(c in df.columns for c in ID_COLS) or any('Naive_Bayes' in c for c in df.columns):
            df = drop_unused_columns(df)

        df = encode_target(df)

        transformed = self._transform_core(df)

        # Ensure all expected columns exist, in the right order
        for col in self.final_feature_columns:
            if col not in transformed.columns:
                transformed[col] = 0

        target = transformed[TARGET_COL] if TARGET_COL in transformed.columns else None
        transformed = transformed[self.final_feature_columns]

        if target is not None:
            transformed[TARGET_COL] = target

        return transformed

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ChurnPreprocessor":
        return joblib.load(path)