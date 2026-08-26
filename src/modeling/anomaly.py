from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


REQUIRED_COLUMNS = {
    "loan_id",
    "reporting_month",
}


def fit_anomaly_model(
    df: pd.DataFrame,
    contamination: float = 0.02,
) -> tuple[IsolationForest, list[str]]:
    """Fit an unsupervised anomaly detector."""

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if not 0 < contamination < 0.5:
        raise ValueError(
            "contamination must be between 0 and 0.5."
        )

    numeric_columns = df.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    excluded = {
        "next_3m_delinquency_flag",
        "next_6m_delinquency_flag",
        "next_12m_default_flag",
        "next_12m_prepayment_flag",
    }

    feature_columns = [
        column
        for column in numeric_columns
        if column not in excluded
    ]

    if not feature_columns:
        raise ValueError(
            "No numeric features available for anomaly detection."
        )

    X = (
        df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X)

    return model, feature_columns


def score_anomalies(
    df: pd.DataFrame,
    model: IsolationForest,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Generate normalized anomaly scores and exception labels."""

    missing = set(feature_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing anomaly features: {sorted(missing)}"
        )

    X = (
        df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    raw_scores = -model.decision_function(X)

    minimum = raw_scores.min()
    maximum = raw_scores.max()

    if maximum > minimum:
        anomaly_scores = (
            raw_scores - minimum
        ) / (maximum - minimum)
    else:
        anomaly_scores = np.zeros(
            len(raw_scores)
        )

    labels = model.predict(X)

    result = df[
        ["loan_id", "reporting_month"]
    ].copy()

    result["anomaly_score"] = anomaly_scores
    result["exception_type"] = np.where(
        labels == -1,
        "ANOMALY",
        "",
    )

    return result


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.02,
) -> pd.DataFrame:
    """Fit and score anomalies on a dataset."""

    model, feature_columns = fit_anomaly_model(
        df,
        contamination=contamination,
    )

    return score_anomalies(
        df,
        model,
        feature_columns,
    )