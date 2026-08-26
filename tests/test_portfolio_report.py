import pandas as pd
import pytest

from src.reporting.portfolio_report import build_portfolio_markdown_report


def _sample_report():
    return {
        "portfolio_summary": {
            "total_observations": 100,
            "unique_loans": 20,
            "flagged_observations": 10,
            "flagged_rate": 0.10,
            "average_predicted_probability": 0.0825,
        },
        "risk_tier_summary": pd.DataFrame(
            {
                "risk_tier": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                "observations": [80, 10, 7, 3],
                "unique_loans": [20, 8, 5, 3],
                "average_probability": [0.03, 0.25, 0.55, 0.85],
                "flagged_observations": [0, 10, 7, 3],
                "observation_share": [0.80, 0.10, 0.07, 0.03],
            }
        ),
        "monthly_risk_trend": pd.DataFrame(
            {
                "reporting_month": pd.to_datetime(
                    ["2024-01-01", "2024-02-01"]
                ),
                "observations": [50, 50],
                "unique_loans": [20, 20],
                "flagged_observations": [4, 6],
                "average_probability": [0.07, 0.095],
                "flagged_rate": [0.08, 0.12],
            }
        ),
        "dimension_summaries": {
            "current_status": pd.DataFrame(
                {
                    "current_status": ["CURRENT", "DELINQUENT"],
                    "observations": [80, 20],
                    "flagged_observations": [3, 7],
                    "average_probability": [0.04, 0.25],
                    "flagged_rate": [0.0375, 0.35],
                }
            ),
        },
    }


def test_build_report_contains_major_sections():
    markdown = build_portfolio_markdown_report(
        _sample_report()
    )

    assert "# Loan Performance Intelligence Report" in markdown
    assert "## Executive Summary" in markdown
    assert "## Risk Tier Distribution" in markdown
    assert "## Monthly Risk Trend" in markdown
    assert "## Risk by Current Status" in markdown


def test_report_contains_formatted_summary_values():
    markdown = build_portfolio_markdown_report(
        _sample_report()
    )

    assert "100" in markdown
    assert "20" in markdown
    assert "10.00%" in markdown
    assert "0.0825" in markdown


def test_report_formats_monthly_dates():
    markdown = build_portfolio_markdown_report(
        _sample_report()
    )

    assert "2024-01-01" in markdown
    assert "2024-02-01" in markdown


def test_report_can_be_written_to_file(tmp_path):
    output_path = tmp_path / "portfolio_report.md"

    markdown = build_portfolio_markdown_report(
        _sample_report(),
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.read_text(
        encoding="utf-8"
    ) == markdown


def test_missing_report_section_raises():
    report = _sample_report()
    del report["risk_tier_summary"]

    with pytest.raises(ValueError):
        build_portfolio_markdown_report(report)