import pandas as pd
import pytest

from src.modeling.portfolio_intelligence import (
    build_dimension_risk_summary,
    build_monthly_risk_trend,
    build_portfolio_intelligence_report,
    build_portfolio_summary,
    build_risk_tier_summary,
)


def _sample_risk_output() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": [
                "L1",
                "L2",
                "L3",
                "L1",
            ],
            "reporting_month": [
                "2024-01-01",
                "2024-01-01",
                "2024-02-01",
                "2024-02-01",
            ],
            "predicted_probability": [
                0.05,
                0.25,
                0.75,
                0.45,
            ],
            "risk_flag": [
                0,
                1,
                1,
                1,
            ],
            "risk_tier": [
                "LOW",
                "MEDIUM",
                "CRITICAL",
                "HIGH",
            ],
            "action_priority": [
                "ROUTINE",
                "MONITOR",
                "IMMEDIATE_REVIEW",
                "REVIEW",
            ],
            "current_status": [
                "CURRENT",
                "CURRENT",
                "DEFAULT",
                "DELINQUENT",
            ],
        }
    )


def test_build_portfolio_summary():
    result = build_portfolio_summary(
        _sample_risk_output()
    )

    assert result["total_observations"] == 4
    assert result["unique_loans"] == 3
    assert result["flagged_observations"] == 3
    assert result["flagged_rate"] == 0.75
    assert result["average_predicted_probability"] == pytest.approx(
        0.375
    )


def test_build_risk_tier_summary():
    result = build_risk_tier_summary(
        _sample_risk_output()
    )

    assert set(result["risk_tier"]) == {
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }

    assert result["observations"].sum() == 4
    assert result["flagged_observations"].sum() == 3


def test_build_monthly_risk_trend():
    result = build_monthly_risk_trend(
        _sample_risk_output()
    )

    assert len(result) == 2
    assert result.loc[0, "reporting_month"] == pd.Timestamp(
        "2024-01-01"
    )
    assert result.loc[0, "flagged_observations"] == 1
    assert result.loc[0, "flagged_rate"] == 0.5

    assert result.loc[1, "flagged_observations"] == 2


def test_build_dimension_risk_summary():
    result = build_dimension_risk_summary(
        _sample_risk_output(),
        "current_status",
    )

    assert set(result["current_status"]) == {
        "CURRENT",
        "DEFAULT",
        "DELINQUENT",
    }

    assert "flagged_rate" in result.columns
    assert "average_probability" in result.columns


def test_invalid_dimension_raises():
    with pytest.raises(ValueError):
        build_dimension_risk_summary(
            _sample_risk_output(),
            "does_not_exist",
        )


def test_missing_required_column_raises():
    data = _sample_risk_output().drop(
        columns=["risk_tier"]
    )

    with pytest.raises(ValueError):
        build_portfolio_summary(data)
        
        
def test_build_portfolio_intelligence_report():
    data = _sample_risk_output()

    feature_data = pd.DataFrame(
        {
            "current_status": [
                "CURRENT",
                "CURRENT",
                "DEFAULT",
                "DELINQUENT",
            ],
            "credit_score_band": [
                "780+",
                "700-739",
                "<620",
                "660-699",
            ],
            "ltv_band": [
                "<=60",
                "71-80",
                ">95",
                "81-90",
            ],
            "dti_band": [
                "<=20",
                "31-36",
                ">50",
                "37-43",
            ],
        }
    )

    report = build_portfolio_intelligence_report(
        data,
        feature_data,
    )

    assert "portfolio_summary" in report
    assert "risk_tier_summary" in report
    assert "monthly_risk_trend" in report
    assert "dimension_summaries" in report

    assert set(
        report["dimension_summaries"].keys()
    ) == {
        "current_status",
        "credit_score_band",
        "ltv_band",
        "dti_band",
    }
    
def test_build_portfolio_intelligence_report_rejects_missing_dimension():
    data = _sample_risk_output()

    feature_data = pd.DataFrame(
        {
            "current_status": [
                "CURRENT",
                "CURRENT",
                "DEFAULT",
                "DELINQUENT",
            ]
        }
    )

    with pytest.raises(ValueError):
        build_portfolio_intelligence_report(
            data,
            feature_data,
        )