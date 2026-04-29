"""
Reusable utilities for the Customer Churn Risk Model.

This module intentionally contains only reusable logic that can support
notebook experimentation or future productionization. Exploratory analysis,
model comparison, and narrative reporting belong in the notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
TARGET_COLUMN = "Exited"


def load_churn_data(path: str | Path = "data/Churn.csv") -> pd.DataFrame:
    """Load the churn dataset from a local CSV path.

    Parameters
    ----------
    path:
        Relative or absolute path to ``Churn.csv``.

    Returns
    -------
    pandas.DataFrame
        Loaded customer-level churn dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty or cannot be parsed into rows.
    """
    data_path = Path(path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Place Churn.csv under data/."
        )

    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError(f"Dataset at {data_path} is empty.")

    return df


def split_features_target(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    drop_columns: Iterable[str] = ID_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features and target while dropping identifier columns."""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' was not found in the dataset.")

    cols_to_drop = [target, *[col for col in drop_columns if col in df.columns]]
    X = df.drop(columns=cols_to_drop)
    y = df[target].astype(int)

    return X, y


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numerical and categorical feature names from a feature matrix."""
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numerical_features]

    return numerical_features, categorical_features


def build_ohe_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build a preprocessing transformer using one-hot encoding."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_ordinal_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build a preprocessing transformer using ordinal encoding.

    This is useful for SMOTE-NC pipelines where categorical indices must be known.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def categorical_indices_after_preprocessing(
    numerical_features: list[str],
    categorical_features: list[str],
) -> list[int]:
    """Return categorical column positions after numeric + categorical preprocessing."""
    start = len(numerical_features)
    stop = start + len(categorical_features)
    return list(range(start, stop))


def best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Find the probability threshold that maximizes F1-score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_values = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )

    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def predict_with_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """Convert churn probabilities into binary predictions using a custom threshold."""
    return (y_proba >= threshold).astype(int)


def f1_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> float:
    """Calculate F1-score at a chosen probability threshold."""
    y_pred = predict_with_threshold(y_proba, threshold)
    return float(f1_score(y_true, y_pred))
