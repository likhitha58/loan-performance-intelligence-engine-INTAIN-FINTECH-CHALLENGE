from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_intelligence.features import (
    build_features,
    get_feature_columns,
)
from src.data_intelligence.loader import load_data_pack
from src.modeling.split import temporal_split


DEFAULT_TARGET = "next_12m_default_flag"


@dataclass
class BaselineResult:
    target: str
    train_rows: int
    validation_rows: int
    positive_rate: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float


def build_baseline_pipeline(
    X: pd.DataFrame,
) -> Pipeline:
    """Build preprocessing + logistic regression baseline."""

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def prepare_model_data(
    df: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build leakage-safe features and return X/y."""

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found."
        )

    features = build_features(df)
    feature_columns = get_feature_columns(features)

    X = features[feature_columns].copy()
    y = features[target].astype(int).copy()

    return X, y


def evaluate_binary_predictions(
    y_true: pd.Series,
    probabilities,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate baseline classification metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }


def run_baseline(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    validation_start: str = "2023-07-01",
) -> BaselineResult:
    """Train and evaluate the chronological logistic baseline."""

    train_df, validation_df = temporal_split(
        df,
        validation_start=validation_start,
    )

    X_train, y_train = prepare_model_data(
        train_df,
        target,
    )

    X_validation, y_validation = prepare_model_data(
        validation_df,
        target,
    )

    pipeline = build_baseline_pipeline(X_train)

    pipeline.fit(
        X_train,
        y_train,
    )

    probabilities = pipeline.predict_proba(
        X_validation
    )[:, 1]

    metrics = evaluate_binary_predictions(
        y_validation,
        probabilities,
    )

    return BaselineResult(
        target=target,
        train_rows=len(X_train),
        validation_rows=len(X_validation),
        positive_rate=float(y_train.mean()),
        **metrics,
    )


def run_default_baseline() -> BaselineResult:
    """Run the default 12-month default baseline."""

    data = load_data_pack()

    return run_baseline(
        data["train"],
        target=DEFAULT_TARGET,
    )