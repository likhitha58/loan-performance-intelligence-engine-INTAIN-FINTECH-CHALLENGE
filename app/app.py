from __future__ import annotations

import streamlit as st

from src.data_intelligence.loader import load_data_pack
from src.data_intelligence.features import build_features
from src.modeling.gradient_boosting import fit_gradient_boosting
from src.modeling.prediction import generate_predictions
from src.modeling.risk_intelligence import add_risk_evidence
from src.modeling.risk_intelligence_output import (
    build_risk_intelligence_output,
)
from src.modeling.portfolio_intelligence import (
    build_portfolio_intelligence_report,
)
from src.modeling.explainability import build_explanation_summary


st.set_page_config(
    page_title="Loan Performance Intelligence",
    layout="wide",
)

st.title("Loan Performance Intelligence Engine")
st.caption("Reviewer Risk Intelligence Dashboard")


@st.cache_data
def load_risk_output():
    data = load_data_pack()

    pipeline, feature_columns = fit_gradient_boosting(
        data["train"],
        target="next_12m_default_flag",
    )

    predictions = generate_predictions(
        data["train"],
        pipeline,
        feature_columns,
        threshold=0.20,
    )

    features = build_features(data["train"])

    evidence = add_risk_evidence(
        predictions,
        features,
    )

    risk_output = build_risk_intelligence_output(
        evidence
    )

    report = build_portfolio_intelligence_report(
        risk_output,
        features,
    )

    return risk_output, features, report


risk_output, features, report = load_risk_output()

summary = report["portfolio_summary"]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Observations",
    f"{summary['total_observations']:,}",
)

col2.metric(
    "Unique Loans",
    f"{summary['unique_loans']:,}",
)

col3.metric(
    "Flagged Observations",
    f"{summary['flagged_observations']:,}",
)

col4.metric(
    "Flagged Rate",
    f"{summary['flagged_rate']:.2%}",
)

st.divider()

st.subheader("Risk Tier Distribution")

tier_summary = report["risk_tier_summary"]

st.dataframe(
    tier_summary,
    use_container_width=True,
    hide_index=True,
)

st.subheader("Reviewer Queue")

flagged = risk_output[
    risk_output["risk_flag"] == 1
].copy()

st.dataframe(
    flagged[
        [
            "loan_id",
            "reporting_month",
            "predicted_probability",
            "risk_tier",
            "action_priority",
            "evidence_category",
        ]
    ].head(100),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Loan Explanation")

if not flagged.empty:
    selected_loan = st.selectbox(
        "Select a flagged loan",
        flagged["loan_id"].unique(),
    )

    selected = flagged[
        flagged["loan_id"] == selected_loan
    ].iloc[0]

    feature_row = features[
        features["loan_id"] == selected_loan
    ]

    if not feature_row.empty:
        row = selected.copy()

        for column in [
            "current_status",
            "credit_score_band",
            "ltv_band",
            "dti_band",
            "dpd_lag_1m",
            "status_change_flag",
            "modification_flag",
        ]:
            if column in feature_row.columns:
                row[column] = feature_row.iloc[0][column]

        explanation = build_explanation_summary(
            row.to_frame().T
        ).iloc[0]

        st.write(
            f"**Risk Tier:** {explanation['risk_tier']}"
        )
        st.write(
            f"**Action:** {explanation['action_priority']}"
        )
        st.write(
            f"**Evidence Category:** "
            f"{explanation['evidence_category']}"
        )

        st.write("**Reasons:**")

        for reason in explanation["reasons"]:
            st.write(f"- {reason}")

        st.info(explanation["explanation"])