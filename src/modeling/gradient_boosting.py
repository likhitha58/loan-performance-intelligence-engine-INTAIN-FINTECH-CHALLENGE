from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.data_intelligence.features import build_features, get_feature_columns
from src.modeling.split import temporal_split


@dataclass
class GradientBoostingResult:
    target: str
    train_rows: int
    validation_rows: int
    positive_rate: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    y_validation: np.ndarray
    validation_probabilities: np.ndarray


def _prepare_data(
    train: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    if target not in train.columns:
        raise ValueError(f"Unknown target: {target}")

    train_part, validation_part = temporal_split(train)

    train_features = build_features(train_part)
    validation_features = build_features(validation_part)

    feature_columns = get_feature_columns(train_features)

    X_train = train_features[feature_columns].copy()
    X_validation = validation_features[feature_columns].copy()

    y_train = train_part[target].astype(int)
    y_validation = validation_part[target].astype(int)

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
    )


def _build_pipeline(
    X: pd.DataFrame,
) -> Pipeline:

    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numeric_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    numeric_pipeline = SimpleImputer(
        strategy="median",
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
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

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def run_gradient_boosting(
    train: pd.DataFrame,
    target: str = "next_12m_default_flag",
) -> GradientBoostingResult:

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = _prepare_data(train, target)

    pipeline = _build_pipeline(X_train)

    pipeline.fit(
        X_train,
        y_train,
    )

    probabilities = pipeline.predict_proba(
        X_validation
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return GradientBoostingResult(
        target=target,
        train_rows=len(X_train),
        validation_rows=len(X_validation),
        positive_rate=float(y_validation.mean()),
        roc_auc=float(
            roc_auc_score(
                y_validation,
                probabilities,
            )
        ),
        pr_auc=float(
            average_precision_score(
                y_validation,
                probabilities,
            )
        ),
        precision=float(
            precision_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        f1=float(
            f1_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        y_validation=y_validation.to_numpy(),
        validation_probabilities=probabilities,
    )