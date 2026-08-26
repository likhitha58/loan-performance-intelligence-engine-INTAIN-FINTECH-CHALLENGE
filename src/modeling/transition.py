from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


STATE_CLASSES = (
    "CURRENT",
    "CLOSED",
    "DELINQUENT",
    "PREPAID",
    "DEFAULT",
)

REQUIRED_COLUMNS = {
    "next_state",
    "reporting_month",
}


def _prepare_transition_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = df.copy()
    data["reporting_month"] = pd.to_datetime(
        data["reporting_month"]
    )

    data = data.sort_values(
        "reporting_month"
    ).reset_index(drop=True)

    y = data["next_state"].astype(str)

    invalid_states = set(y.unique()) - set(STATE_CLASSES)

    if invalid_states:
        raise ValueError(
            f"Unexpected next_state values: "
            f"{sorted(invalid_states)}"
        )

    excluded = {
        "next_state",
        "reporting_month",
        "loan_id",
        "next_3m_delinquency_flag",
        "next_6m_delinquency_flag",
        "next_12m_default_flag",
        "next_12m_prepayment_flag",
    }

    feature_columns = [
        column
        for column in data.columns
        if column not in excluded
    ]

    if not feature_columns:
        raise ValueError(
            "No usable transition features found."
        )

    X = data[feature_columns].copy()

    return X, y, feature_columns


def _build_pipeline(
    X: pd.DataFrame,
) -> Pipeline:
    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        )
                    ]
                ),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    model = GradientBoostingClassifier(
        random_state=42
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def fit_transition_model(
    df: pd.DataFrame,
    validation_fraction: float = 0.20,
) -> tuple[Pipeline, list[str], dict[str, float]]:
    """
    Fit a chronological multiclass next-state model.

    The latest observations are reserved for validation.
    """

    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    X, y, feature_columns = _prepare_transition_data(df)

    split_index = int(
        len(X) * (1 - validation_fraction)
    )

    if split_index <= 0 or split_index >= len(X):
        raise ValueError(
            "Validation split produced an empty partition."
        )

    X_train = X.iloc[:split_index]
    X_validation = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_validation = y.iloc[split_index:]

    pipeline = _build_pipeline(X_train)

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(
        X_validation
    )

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_validation,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_validation,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }

    return pipeline, feature_columns, metrics


def predict_next_state(
    df: pd.DataFrame,
    pipeline: Pipeline,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Generate next-state predictions and class probabilities."""

    missing = set(feature_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required feature columns: "
            f"{sorted(missing)}"
        )

    X = df[feature_columns].copy()

    predictions = pipeline.predict(X)

    probabilities = pipeline.predict_proba(X)

    result = pd.DataFrame(
        {
            "pred_next_state": predictions,
        },
        index=df.index,
    )

    classes = pipeline.classes_

    for index, state in enumerate(classes):
        result[
            f"prob_next_state_{str(state).lower()}"
        ] = probabilities[:, index]

    result["next_state_confidence"] = (
        probabilities.max(axis=1)
    )

    return result


def evaluate_transition_model(
    df: pd.DataFrame,
    pipeline: Pipeline,
    feature_columns: list[str],
) -> dict[str, float]:
    """Evaluate a fitted transition model on supplied data."""

    if "next_state" not in df.columns:
        raise ValueError(
            "next_state column is required."
        )

    predictions = predict_next_state(
        df,
        pipeline,
        feature_columns,
    )

    y_true = df["next_state"].astype(str)

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions["pred_next_state"],
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions["pred_next_state"],
                average="macro",
                zero_division=0,
            )
        ),
    }