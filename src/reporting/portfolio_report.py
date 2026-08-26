from __future__ import annotations

from pathlib import Path
from typing import Any


def _format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_number(value: float) -> str:
    return f"{value:,.0f}"


def build_portfolio_markdown_report(
    report: dict[str, Any],
    output_path: str | Path | None = None,
) -> str:
    """
    Convert portfolio intelligence outputs into a reviewer-friendly
    Markdown report.

    The function does not calculate new model metrics. It only formats
    the already-computed portfolio intelligence results.
    """
    required_sections = {
        "portfolio_summary",
        "risk_tier_summary",
        "monthly_risk_trend",
        "dimension_summaries",
    }

    missing = required_sections - set(report)

    if missing:
        raise ValueError(
            f"Missing required report sections: {sorted(missing)}"
        )

    summary = report["portfolio_summary"]
    risk_tiers = report["risk_tier_summary"]
    monthly_trend = report["monthly_risk_trend"]
    dimensions = report["dimension_summaries"]

    lines: list[str] = []

    lines.extend(
        [
            "# Loan Performance Intelligence Report",
            "",
            "## Executive Summary",
            "",
            (
                f"- **Total observations:** "
                f"{_format_number(summary['total_observations'])}"
            ),
            (
                f"- **Unique loans:** "
                f"{_format_number(summary['unique_loans'])}"
            ),
            (
                f"- **Flagged observations:** "
                f"{_format_number(summary['flagged_observations'])}"
            ),
            (
                f"- **Flagged rate:** "
                f"{_format_percentage(summary['flagged_rate'])}"
            ),
            (
                f"- **Average predicted probability:** "
                f"{summary['average_predicted_probability']:.4f}"
            ),
            "",
            "## Risk Tier Distribution",
            "",
            "| Risk Tier | Observations | Unique Loans | Average Probability | Flagged Observations | Share |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in risk_tiers.iterrows():
        lines.append(
            "| "
            f"{row['risk_tier']} | "
            f"{row['observations']:,} | "
            f"{row['unique_loans']:,} | "
            f"{row['average_probability']:.4f} | "
            f"{row['flagged_observations']:,} | "
            f"{_format_percentage(row['observation_share'])} |"
        )

    lines.extend(
        [
            "",
            "## Monthly Risk Trend",
            "",
            "| Reporting Month | Observations | Flagged | Average Probability | Flagged Rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for _, row in monthly_trend.iterrows():
        month = row["reporting_month"]

        if hasattr(month, "strftime"):
            month = month.strftime("%Y-%m-%d")

        lines.append(
            "| "
            f"{month} | "
            f"{row['observations']:,} | "
            f"{row['flagged_observations']:,} | "
            f"{row['average_probability']:.4f} | "
            f"{_format_percentage(row['flagged_rate'])} |"
        )

    for dimension, dataframe in dimensions.items():
        lines.extend(
            [
                "",
                f"## Risk by {dimension.replace('_', ' ').title()}",
                "",
            ]
        )

        columns = list(dataframe.columns)

        display_columns = [
            column
            for column in columns
            if column != dimension
        ]

        lines.append(
            "| "
            + dimension.replace("_", " ").title()
            + " | "
            + " | ".join(
                column.replace("_", " ").title()
                for column in display_columns
            )
            + " |"
        )

        lines.append(
            "|---|"
            + "---:|" * len(display_columns)
        )

        for _, row in dataframe.iterrows():
            values = []

            for column in display_columns:
                value = row[column]

                if column == "flagged_rate":
                    value = _format_percentage(value)
                elif column == "average_probability":
                    value = f"{value:.4f}"
                elif isinstance(value, (int, float)):
                    value = f"{value:,}"

                values.append(str(value))

            lines.append(
                "| "
                + str(row[dimension])
                + " | "
                + " | ".join(values)
                + " |"
            )

    markdown = "\n".join(lines) + "\n"

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")

    return markdown