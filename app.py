from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Loan Performance Intelligence Engine",
    page_icon="📊",
    layout="wide",
)


ROOT = Path(__file__).resolve().parent
SUBMISSION_DIR = ROOT / "submission"


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

@st.cache_data
def load_submission() -> pd.DataFrame:
    path = SUBMISSION_DIR / "submission.csv"

    if not path.exists():
        raise FileNotFoundError(
            "submission/submission.csv was not found. "
            "Run: python -m scripts.generate_submission"
        )

    df = pd.read_csv(path)

    if "reporting_month" in df.columns:
        df["reporting_month"] = pd.to_datetime(
            df["reporting_month"],
            errors="coerce",
        )

    return df


@st.cache_data
def load_global_importance() -> pd.DataFrame:
    path = SUBMISSION_DIR / "global_feature_importance.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_scenario_summary() -> pd.DataFrame:
    path = SUBMISSION_DIR / "scenario_summary.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_segment_impacts() -> pd.DataFrame:
    path = SUBMISSION_DIR / "scenario_segment_impacts.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def probability_mean(
    df: pd.DataFrame,
    column: str,
) -> float:
    if column not in df.columns or df.empty:
        return 0.0

    return float(
        pd.to_numeric(
            df[column],
            errors="coerce",
        ).mean()
    )


def format_probability(value: float) -> str:
    return f"{value:.1%}"


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

try:
    submission = load_submission()
    global_importance = load_global_importance()
    scenario_summary = load_scenario_summary()
    segment_impacts = load_segment_impacts()

except Exception as exc:
    st.error(str(exc))
    st.stop()


# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------

st.title("Loan Performance Intelligence Engine")

st.caption(
    "Portfolio risk intelligence, explainability, scenario analysis, "
    "and reviewer-oriented decision support."
)


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------

st.sidebar.header("Navigation")

section = st.sidebar.radio(
    "Go to",
    [
        "Portfolio Overview",
        "Loan Explorer",
        "Global Explainability",
        "Scenario Analysis",
    ],
)


# ---------------------------------------------------------------------
# PORTFOLIO OVERVIEW
# ---------------------------------------------------------------------

if section == "Portfolio Overview":

    st.header("Portfolio Overview")

    total_observations = len(submission)

    anomaly_count = 0
    if "anomaly_score" in submission.columns:
        anomaly_scores = pd.to_numeric(
            submission["anomaly_score"],
            errors="coerce",
        )
        anomaly_count = int(
            (anomaly_scores > 0).sum()
        )

    avg_3m = probability_mean(
        submission,
        "pred_next_3m_delinquency_prob",
    )

    avg_6m = probability_mean(
        submission,
        "pred_next_6m_delinquency_prob",
    )

    avg_default = probability_mean(
        submission,
        "pred_next_12m_default_prob",
    )

    avg_prepayment = probability_mean(
        submission,
        "pred_next_12m_prepayment_prob",
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Loan-month observations",
        f"{total_observations:,}",
    )

    c2.metric(
        "Avg 3M delinquency",
        format_probability(avg_3m),
    )

    c3.metric(
        "Avg 12M default",
        format_probability(avg_default),
    )

    c4.metric(
        "Anomalous observations",
        f"{anomaly_count:,}",
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Average Event Probabilities")

        probability_data = pd.DataFrame(
            {
                "Event": [
                    "3M Delinquency",
                    "6M Delinquency",
                    "12M Default",
                    "12M Prepayment",
                ],
                "Probability": [
                    avg_3m,
                    avg_6m,
                    avg_default,
                    avg_prepayment,
                ],
            }
        )

        st.bar_chart(
            probability_data.set_index("Event")
        )

    with right:

        st.subheader("Predicted Next-State Distribution")

        if "pred_next_state" in submission.columns:

            state_counts = (
                submission["pred_next_state"]
                .value_counts()
                .rename_axis("State")
                .to_frame("Observations")
            )

            st.bar_chart(state_counts)

        else:
            st.info(
                "Next-state predictions are not available."
            )

    st.divider()

    st.subheader("Recommended Action Distribution")

    if "recommended_action" in submission.columns:

        action_counts = (
            submission["recommended_action"]
            .value_counts()
            .rename_axis("Action")
            .to_frame("Observations")
        )

        st.dataframe(
            action_counts,
            use_container_width=True,
        )

    else:
        st.info(
            "Recommended-action information is not available."
        )


# ---------------------------------------------------------------------
# LOAN EXPLORER
# ---------------------------------------------------------------------

elif section == "Loan Explorer":

    st.header("Loan Explorer")

    loan_ids = (
        submission["loan_id"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
    )

    selected_loan = st.selectbox(
        "Select loan",
        loan_ids,
    )

    loan_rows = submission[
        submission["loan_id"].astype(str)
        == selected_loan
    ].copy()

    if loan_rows.empty:
        st.warning("No observations found for this loan.")
        st.stop()

    selected_month = st.selectbox(
        "Select reporting month",
        sorted(
            loan_rows["reporting_month"]
            .dropna()
            .dt.strftime("%Y-%m-%d")
            .unique(),
            reverse=True,
        ),
    )

    row = loan_rows[
        loan_rows["reporting_month"]
        .dt.strftime("%Y-%m-%d")
        == selected_month
    ].iloc[0]

    st.subheader(
        f"Loan {selected_loan} — {selected_month}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "3M Delinquency",
        format_probability(
            float(row["pred_next_3m_delinquency_prob"])
        ),
    )

    c2.metric(
        "6M Delinquency",
        format_probability(
            float(row["pred_next_6m_delinquency_prob"])
        ),
    )

    c3.metric(
        "12M Default",
        format_probability(
            float(row["pred_next_12m_default_prob"])
        ),
    )

    c4.metric(
        "12M Prepayment",
        format_probability(
            float(row["pred_next_12m_prepayment_prob"])
        ),
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Predicted Next State")

        st.write(
            row.get(
                "pred_next_state",
                "Not available",
            )
        )

        st.subheader("Recommended Action")

        st.write(
            row.get(
                "recommended_action",
                "Not available",
            )
        )

        st.subheader("Confidence")

        confidence = pd.to_numeric(
            row.get("confidence", 0),
            errors="coerce",
        )

        if pd.notna(confidence):
            st.progress(
                min(max(float(confidence), 0.0), 1.0)
            )
            st.write(
                format_probability(float(confidence))
            )

    with right:

        st.subheader("Anomaly Information")

        anomaly_score = pd.to_numeric(
            row.get("anomaly_score", 0),
            errors="coerce",
        )

        if pd.notna(anomaly_score):
            st.metric(
                "Anomaly Score",
                f"{float(anomaly_score):.4f}",
            )

        exception_type = row.get(
            "exception_type",
            "",
        )

        if pd.notna(exception_type) and str(
            exception_type
        ).strip():

            st.warning(
                f"Exception: {exception_type}"
            )

        else:
            st.success(
                "No exception recorded for this observation."
            )

    st.divider()

    st.subheader("Model Drivers / Risk Evidence")

    drivers = row.get(
        "top_drivers",
        "",
    )

    if pd.notna(drivers) and str(drivers).strip():

        st.write(str(drivers))

    else:

        st.info(
            "No explicit top-driver information was recorded "
            "for this observation."
        )


# ---------------------------------------------------------------------
# GLOBAL EXPLAINABILITY
# ---------------------------------------------------------------------

elif section == "Global Explainability":

    st.header("Global Model Explainability")

    if global_importance.empty:

        st.warning(
            "Global feature importance output is unavailable."
        )

    else:

        models = (
            global_importance["model"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_model = st.selectbox(
            "Select model",
            models,
        )

        model_importance = global_importance[
            global_importance["model"]
            == selected_model
        ].copy()

        model_importance = model_importance.sort_values(
            "rank"
        )

        top_n = st.slider(
            "Number of features",
            min_value=5,
            max_value=min(
                20,
                len(model_importance),
            ),
            value=min(
                10,
                len(model_importance),
            ),
        )

        display_data = model_importance.head(
            top_n
        ).copy()

        st.subheader(
            f"Top {top_n} Features"
        )

        chart_data = display_data[
            [
                "feature",
                "importance",
            ]
        ].set_index("feature")

        st.bar_chart(chart_data)

        st.dataframe(
            display_data,
            use_container_width=True,
        )


# ---------------------------------------------------------------------
# SCENARIO ANALYSIS
# ---------------------------------------------------------------------

elif section == "Scenario Analysis":

    st.header("Macro Scenario Analysis")

    if scenario_summary.empty:

        st.warning(
            "Scenario summary output is unavailable."
        )

    else:

        st.subheader("Portfolio Scenario Summary")

        st.dataframe(
            scenario_summary,
            use_container_width=True,
        )

    st.divider()

    if segment_impacts.empty:

        st.warning(
            "Segment scenario impact output is unavailable."
        )

    else:

        st.subheader("Segment-Level Scenario Impacts")

        st.dataframe(
            segment_impacts,
            use_container_width=True,
        )


# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------

st.divider()

st.caption(
    "Decision-support prototype. Model outputs are signals for "
    "human review and should not be treated as autonomous "
    "underwriting decisions."
)
