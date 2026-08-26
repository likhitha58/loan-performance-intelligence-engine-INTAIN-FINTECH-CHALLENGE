from __future__ import annotations

from pathlib import Path

from src.data_intelligence.features import build_features
from src.data_intelligence.loader import load_data_pack
from src.modeling.gradient_boosting import fit_gradient_boosting
from src.modeling.portfolio_intelligence import (
    build_portfolio_intelligence_report,
)
from src.modeling.prediction import generate_predictions
from src.modeling.risk_intelligence import add_risk_evidence
from src.modeling.risk_intelligence_output import (
    build_risk_intelligence_output,
)
from src.reporting.portfolio_report import (
    build_portfolio_markdown_report,
)


TARGET = "next_12m_default_flag"
THRESHOLD = 0.20
OUTPUT_PATH = Path("reports/portfolio_risk_report.md")


def run_pipeline() -> Path:
    data = load_data_pack()

    train = data["train"]

    pipeline, features = fit_gradient_boosting(
        train,
        target=TARGET,
    )

    predictions = generate_predictions(
        train,
        pipeline,
        features,
        threshold=THRESHOLD,
    )

    feature_data = build_features(train)

    evidence = add_risk_evidence(
        predictions,
        feature_data,
    )

    risk_output = build_risk_intelligence_output(
        evidence,
    )

    portfolio_report = build_portfolio_intelligence_report(
        risk_output,
        feature_data,
    )

    build_portfolio_markdown_report(
        portfolio_report,
        output_path=OUTPUT_PATH,
    )

    return OUTPUT_PATH


if __name__ == "__main__":
    output = run_pipeline()
    print(f"Pipeline completed successfully.")
    print(f"Report: {output}")