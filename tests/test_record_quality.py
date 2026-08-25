import pandas as pd

from src.data_intelligence.record_quality import (
    build_record_quality_metrics,
    calculate_record_missingness,
)


def test_record_missingness():

    df = pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "reporting_month": [
                "2024-01-01",
                "2024-02-01",
            ],
            "balance": [100.0, None],
            "rate": [5.0, None],
        }
    )

    result = calculate_record_missingness(df)

    assert result.iloc[0] == 0
    assert result.iloc[1] == 100


def test_build_record_quality_metrics():

    df = pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "reporting_month": [
                "2024-01-01",
                "2024-02-01",
            ],
            "balance": [100.0, None],
        }
    )

    relationship_flags = pd.DataFrame(
        {
            "R003": [False, True],
            "R004": [False, True],
            "R009": [False, False],
        }
    )

    outlier_flags = pd.DataFrame(
        {
            "balance": [False, True],
            "rate": [False, False],
        }
    )

    conflict_flags = pd.Series(
        [False, True]
    )

    result = build_record_quality_metrics(
        df,
        relationship_flags=relationship_flags,
        outlier_flags=outlier_flags,
        conflict_flags=conflict_flags,
    )

    assert len(result) == 2

    assert result.loc[
        0,
        "relationship_violation_rate",
    ] == 0

    assert round(
        result.loc[
            1,
            "relationship_violation_rate",
        ],
        2,
    ) == 66.67

    assert result.loc[
        1,
        "outlier_rate",
    ] == 50

    assert result.loc[
        1,
        "conflict_rate",
    ] == 100


def test_clean_records_have_zero_rates():

    df = pd.DataFrame(
        {
            "loan_id": ["L1"],
            "reporting_month": ["2024-01-01"],
            "balance": [100.0],
        }
    )

    result = build_record_quality_metrics(df)

    assert result.loc[
        0,
        "relationship_violation_rate",
    ] == 0

    assert result.loc[
        0,
        "outlier_rate",
    ] == 0

    assert result.loc[
        0,
        "conflict_rate",
    ] == 0