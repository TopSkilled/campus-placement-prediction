"""
Data Preprocessing Pipeline
Handles encoding, scaling, and train/test splitting for the placement dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

CATEGORICAL_COLS = [
    'gender', 'ssc_board', 'hsc_board', 'hsc_subject',
    'degree_type', 'work_experience', 'mba_specialization'
]
NUMERIC_COLS = [
    'ssc_percentage', 'hsc_percentage', 'degree_percentage',
    'etest_percentage', 'mba_percentage'
]


class PlacementPreprocessor:
    """Handles all preprocessing for the placement dataset."""

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []

    def fit_transform(self, df: pd.DataFrame, target: str = 'placement_status'):
        """
        Fit preprocessor on training data and return transformed arrays.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataset (with salary column included)
        target : str
            Target column name

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        df = df.copy()

        # Drop salary column for classification task (it's a separate target)
        if 'salary' in df.columns:
            df = df.drop(columns=['salary'])

        # Encode categoricals
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        # Separate features and target
        y = df[target].values
        X = df.drop(columns=[target])

        self.feature_columns = list(X.columns)

        # Scale numeric columns
        X[NUMERIC_COLS] = self.scaler.fit_transform(X[NUMERIC_COLS])

        return train_test_split(X.values, y, test_size=0.2, random_state=42, stratify=y)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted encoders and scaler."""
        df = df.copy()

        for col in CATEGORICAL_COLS:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                df[col] = df[col].map(
                    lambda x, le=le: le.transform([x])[0] if x in le.classes_ else 0
                )

        if set(NUMERIC_COLS).issubset(df.columns):
            df[NUMERIC_COLS] = self.scaler.transform(df[NUMERIC_COLS])

        if self.feature_columns:
            df = df[self.feature_columns]

        return df.values

    def save(self, path: str):
        """Persist preprocessor to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: str) -> 'PlacementPreprocessor':
        """Load preprocessor from disk."""
        return joblib.load(path)


def prepare_salary_data(df: pd.DataFrame, preprocessor: PlacementPreprocessor):
    """
    Prepare data for salary regression (placed students only).

    Returns
    -------
    X_train, X_test, y_train, y_test (salary in LPA)
    """
    placed = df[df['placement_status'] == 1].copy()
    salary_target = placed['salary'].values

    # Reuse label encoders, refit scaler on this subset
    placed_clean = placed.drop(columns=['salary', 'placement_status'])
    for col in CATEGORICAL_COLS:
        if col in preprocessor.label_encoders:
            le = preprocessor.label_encoders[col]
            placed_clean[col] = placed_clean[col].map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else 0
            )

    salary_scaler = StandardScaler()
    placed_clean[NUMERIC_COLS] = salary_scaler.fit_transform(placed_clean[NUMERIC_COLS])

    return train_test_split(
        placed_clean.values, salary_target,
        test_size=0.2, random_state=42
    ), salary_scaler
