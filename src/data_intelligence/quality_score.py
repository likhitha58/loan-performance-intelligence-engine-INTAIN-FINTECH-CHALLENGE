from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class QualityScoreConfig:
    missingness_weight: float = 25.0
    relationship_weight: float = 45.0
    outlier_weight: float = 15.0
    conflict_weight: float = 15.0

    good_threshold: float = 90.0
    review_threshold: float = 70.0


def _clip(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def classify_quality(
    score: float,
    config: QualityScoreConfig | None = None,
) -> str:
    config = config or QualityScoreConfig()

    score = _clip(score)

    if score >= config.good_threshold:
        return "GOOD"

    if score >= config.review_threshold:
        return "REVIEW"

    return "CRITICAL"


def calculate_quality_score(
    *,
    missingness_rate: float,
    relationship_violation_rate: float,
    outlier_rate: float,
    conflict_rate: float,
    config: QualityScoreConfig | None = None,
) -> dict:
    """
    Calculate a deterministic data-quality score.

    Input rates are percentages from 0 to 100.
    """

    config = config or QualityScoreConfig()

    missingness_rate = _clip(missingness_rate)
    relationship_violation_rate = _clip(
        relationship_violation_rate
    )
    outlier_rate = _clip(outlier_rate)
    conflict_rate = _clip(conflict_rate)

    missing_penalty = (
        missingness_rate / 100.0
    ) * config.missingness_weight

    relationship_penalty = (
        relationship_violation_rate / 100.0
    ) * config.relationship_weight

    outlier_penalty = (
        outlier_rate / 100.0
    ) * config.outlier_weight

    conflict_penalty = (
        conflict_rate / 100.0
    ) * config.conflict_weight

    total_penalty = (
        missing_penalty
        + relationship_penalty
        + outlier_penalty
        + conflict_penalty
    )

    score = _clip(100.0 - total_penalty)

    return {
        "quality_score": round(score, 2),
        "quality_rating": classify_quality(
            score,
            config,
        ),
        "missingness_penalty": round(
            missing_penalty,
            2,
        ),
        "relationship_penalty": round(
            relationship_penalty,
            2,
        ),
        "outlier_penalty": round(
            outlier_penalty,
            2,
        ),
        "conflict_penalty": round(
            conflict_penalty,
            2,
        ),
        "total_penalty": round(
            total_penalty,
            2,
        ),
    }


def score_dataframe(
    metrics: pd.DataFrame,
    config: QualityScoreConfig | None = None,
) -> pd.DataFrame:
    """
    Calculate quality scores for a dataframe containing
    one row per record.

    Expected columns:

        loan_id
        missingness_rate
        relationship_violation_rate
        outlier_rate
        conflict_rate
    """

    config = config or QualityScoreConfig()

    required = {
        "loan_id",
        "missingness_rate",
        "relationship_violation_rate",
        "outlier_rate",
        "conflict_rate",
    }

    missing = required - set(metrics.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    results = []

    for _, row in metrics.iterrows():

        scored = calculate_quality_score(
            missingness_rate=row["missingness_rate"],
            relationship_violation_rate=(
                row["relationship_violation_rate"]
            ),
            outlier_rate=row["outlier_rate"],
            conflict_rate=row["conflict_rate"],
            config=config,
        )

        results.append(
            {
                "loan_id": row["loan_id"],
                **scored,
            }
        )

    return pd.DataFrame(results)