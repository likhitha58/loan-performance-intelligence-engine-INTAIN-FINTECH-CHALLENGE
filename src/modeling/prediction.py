from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_intelligence.features import build_features


def generate_predictions(
    data: pd.DataFrame,
    pipeline: Pipeline,
    feature_columns: list[str],
    threshold: float = 0.20,
) -> pd.DataFrame:
    """Generate point-in-time risk predictions for supplied observations."""

    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")

    features = build_features(data)
    X = features[feature_columns].copy()

    probabilities = pipeline.predict_proba(X)[:, 1]

    return pd.DataFrame(
        {
            "loan_id": data["loan_id"].to_numpy(),
            "reporting_month": data["reporting_month"].to_numpy(),
            "predicted_probability": probabilities,
            "risk_flag": (
                probabilities >= threshold
            ).astype(int),
            "threshold": threshold,
        }
    )