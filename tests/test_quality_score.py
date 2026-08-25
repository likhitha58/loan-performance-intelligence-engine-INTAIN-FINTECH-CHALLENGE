import pandas as pd
import pytest

from src.data_intelligence.quality_score import (
    QualityScoreConfig,
    calculate_quality_score,
    classify_quality,
    score_dataframe,
)


def test_perfect_record_is_good():

    result = calculate_quality_score(
        missingness_rate=0,
        relationship_violation_rate=0,
        outlier_rate=0,
        conflict_rate=0,
    )

    assert result["quality_score"] == 100
    assert result["quality_rating"] == "GOOD"


def test_quality_score_decreases_with_issues():

    result = calculate_quality_score(
        missingness_rate=20,
        relationship_violation_rate=20,
        outlier_rate=20,
        conflict_rate=20,
    )

    assert result["quality_score"] < 100
    assert result["total_penalty"] > 0


def test_critical_classification():

    assert classify_quality(69) == "CRITICAL"
    assert classify_quality(70) == "REVIEW"
    assert classify_quality(89) == "REVIEW"
    assert classify_quality(90) == "GOOD"


def test_score_dataframe():

    metrics = pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "missingness_rate": [0, 20],
            "relationship_violation_rate": [0, 20],
            "outlier_rate": [0, 0],
            "conflict_rate": [0, 0],
        }
    )

    result = score_dataframe(metrics)

    assert len(result) == 2
    assert result.loc[0, "quality_score"] == 100
    assert result.loc[1, "quality_score"] < 100


def test_missing_columns_raise_error():

    metrics = pd.DataFrame(
        {
            "loan_id": ["L1"],
            "missingness_rate": [0],
        }
    )

    with pytest.raises(ValueError):
        score_dataframe(metrics)


def test_custom_configuration():

    config = QualityScoreConfig(
        missingness_weight=50,
        relationship_weight=20,
        outlier_weight=20,
        conflict_weight=10,
    )

    result = calculate_quality_score(
        missingness_rate=10,
        relationship_violation_rate=0,
        outlier_rate=0,
        conflict_rate=0,
        config=config,
    )

    assert result["missingness_penalty"] == 5
    assert result["quality_score"] == 95