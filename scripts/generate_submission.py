from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_intelligence.features import build_features
from src.data_intelligence.loader import load_data_pack
from src.modeling.anomaly import detect_anomalies
from src.modeling.gradient_boosting import fit_gradient_boosting
from src.modeling.prediction import generate_predictions
from src.modeling.risk_intelligence import add_risk_evidence
from src.modeling.risk_intelligence_output import (
    build_risk_intelligence_output,
)
from src.modeling.scenario_simulation import (
    run_all_macro_scenarios,
    summarize_macro_scenario,
    summarize_segment_impacts,
)
from src.modeling.transition import (
    fit_transition_model,
    predict_next_state,
)
from src.submission import (
    build_submission,
    write_submission,
)


# ---------------------------------------------------------------------
# Prediction targets
# ---------------------------------------------------------------------

TARGETS = {
    "next_3m_delinquency_flag":
        "pred_next_3m_delinquency_prob",

    "next_6m_delinquency_flag":
        "pred_next_6m_delinquency_prob",

    "next_12m_default_flag":
        "pred_next_12m_default_prob",

    "next_12m_prepayment_flag":
        "pred_next_12m_prepayment_prob",
}


RISK_THRESHOLD = 0.20


SEGMENT_COLUMNS = [
    "credit_score_band",
    "vintage",
    "servicer",
    "loan_state",
]


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main() -> None:

    # ================================================================
    # 1. Load data
    # ================================================================

    print("=" * 70)
    print("LOAN PERFORMANCE INTELLIGENCE ENGINE")
    print("=" * 70)

    print("\nLoading data...")

    data = load_data_pack()

    train = data["train"]
    test = data["test"]

    print(f"  Train rows: {len(train)}")
    print(f"  Test rows:  {len(test)}")

    # Ensure submission directory exists.
    submission_dir = Path("submission")
    submission_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Base dataframe containing submission identifiers.
    prediction_data = test[
        [
            "loan_id",
            "reporting_month",
        ]
    ].copy()

    # ================================================================
    # 2. Train probability models
    # ================================================================

    print("\n" + "-" * 70)
    print("TRAINING EVENT PREDICTION MODELS")
    print("-" * 70)

    for target, output_column in TARGETS.items():

        print(f"\nTraining {target}...")

        pipeline, features = fit_gradient_boosting(
            train,
            target=target,
        )

        predictions = generate_predictions(
            test,
            pipeline,
            features,
            threshold=RISK_THRESHOLD,
        )

        prediction_data[output_column] = (
            predictions[
                "predicted_probability"
            ].to_numpy()
        )

        print(
            f"  Predictions generated: "
            f"{len(predictions)}"
        )

        print(
            f"  Features used: "
            f"{len(features)}"
        )

    # ================================================================
    # 3. Macro scenario simulation
    # ================================================================

    print("\n" + "-" * 70)
    print("RUNNING MACRO SCENARIO ANALYSIS")
    print("-" * 70)

    # Combine predictions with the original test attributes.
    #
    # The scenario engine needs both:
    #   - model probabilities
    #   - portfolio/loan attributes
    #
    scenario_predictions = prediction_data.merge(
        test,
        on=[
            "loan_id",
            "reporting_month",
        ],
        how="left",
        suffixes=(
            "",
            "_source",
        ),
    )

    scenario_outputs = run_all_macro_scenarios(
        scenario_predictions,
        segment_columns=SEGMENT_COLUMNS,
    )

    # ---------------------------------------------------------------
    # 3A. Portfolio-level scenario summary
    # ---------------------------------------------------------------

    scenario_summary_rows = []

    for scenario_name, scenario_output in (
        scenario_outputs.items()
    ):

        summary = summarize_macro_scenario(
            scenario_output
        )

        scenario_summary_rows.append(
            summary
        )

    scenario_summary = pd.DataFrame(
        scenario_summary_rows
    )

    scenario_summary_path = (
        submission_dir
        / "scenario_summary.csv"
    )

    scenario_summary.to_csv(
        scenario_summary_path,
        index=False,
    )

    print(
        f"\n  Portfolio scenario report: "
        f"{scenario_summary_path}"
    )

    # ---------------------------------------------------------------
    # 3B. Segment-level scenario impacts
    # ---------------------------------------------------------------

    segment_rows = []

    for scenario_name, scenario_output in (
        scenario_outputs.items()
    ):

        for segment_column in SEGMENT_COLUMNS:

            if segment_column not in scenario_output.columns:
                continue

            segment_result = (
                summarize_segment_impacts(
                    scenario_output,
                    segment_column,
                )
            )

            # Add scenario metadata.
            segment_result.insert(
                0,
                "scenario",
                scenario_name,
            )

            segment_result.insert(
                1,
                "segment_type",
                segment_column,
            )

            segment_rows.append(
                segment_result
            )

    if segment_rows:

        scenario_segment_impacts = pd.concat(
            segment_rows,
            ignore_index=True,
        )

        scenario_segment_path = (
            submission_dir
            / "scenario_segment_impacts.csv"
        )

        scenario_segment_impacts.to_csv(
            scenario_segment_path,
            index=False,
        )

        print(
            f"  Segment scenario report: "
            f"{scenario_segment_path}"
        )

    # ================================================================
    # 4. Next-state / transition model
    # ================================================================

    print("\n" + "-" * 70)
    print("TRAINING NEXT-STATE TRANSITION MODEL")
    print("-" * 70)

    transition_pipeline, transition_features, metrics = (
        fit_transition_model(train)
    )

    print(
        f"  Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"  Macro F1: "
        f"{metrics['macro_f1']:.4f}"
    )

    transition_predictions = predict_next_state(
        test,
        transition_pipeline,
        transition_features,
    )

    transition_data = test[
        [
            "loan_id",
            "reporting_month",
        ]
    ].copy()

    transition_data["pred_next_state"] = (
        transition_predictions[
            "pred_next_state"
        ].to_numpy()
    )

    transition_data["next_state_confidence"] = (
        transition_predictions[
            "next_state_confidence"
        ].to_numpy()
    )

    # ================================================================
    # 5. Anomaly detection
    # ================================================================

    print("\n" + "-" * 70)
    print("DETECTING ANOMALIES")
    print("-" * 70)

    anomaly_data = detect_anomalies(
        test
    )

    anomaly_count = int(
        (
            anomaly_data[
                "exception_type"
            ] != ""
        ).sum()
    )

    print(
        f"  Anomalies detected: "
        f"{anomaly_count}"
    )

    print(
        f"  Normal observations: "
        f"{len(anomaly_data) - anomaly_count}"
    )

    # ================================================================
    # 6. Risk evidence and explainability
    # ================================================================

    print("\n" + "-" * 70)
    print("BUILDING RISK EVIDENCE")
    print("-" * 70)

    test_features = build_features(
        test
    )

    # The risk intelligence module currently
    # uses the 12-month default probability.
    risk_predictions = prediction_data[
        [
            "loan_id",
            "reporting_month",
            "pred_next_12m_default_prob",
        ]
    ].copy()

    risk_predictions = risk_predictions.rename(
        columns={
            "pred_next_12m_default_prob":
                "predicted_probability"
        }
    )

    risk_predictions["risk_flag"] = (
        risk_predictions[
            "predicted_probability"
        ]
        >= RISK_THRESHOLD
    )

    risk_predictions["threshold"] = (
        RISK_THRESHOLD
    )

    risk_evidence = add_risk_evidence(
        risk_predictions,
        test_features,
    )

    risk_output = (
        build_risk_intelligence_output(
            risk_evidence
        )
    )

    # Normalize the evidence into the
    # "reasons" field expected by submission.py.
    risk_output["reasons"] = (
        risk_output[
            "risk_evidence"
        ].apply(
            lambda values:
                values
                if isinstance(values, list)
                else []
        )
    )

    risk_output = risk_output[
        [
            "loan_id",
            "reporting_month",
            "reasons",
            "action_priority",
        ]
    ]

    print(
        f"  Risk evidence generated for "
        f"{len(risk_output)} observations"
    )

    # ================================================================
    # 7. Build final submission
    # ================================================================

    print("\n" + "-" * 70)
    print("BUILDING FINAL SUBMISSION")
    print("-" * 70)

    submission = build_submission(
        prediction_data,
        transition_data=transition_data,
        anomaly_data=anomaly_data,
        explanation_data=risk_output,
    )

    output_path = (
        submission_dir
        / "submission.csv"
    )

    write_submission(
        submission,
        output_path,
    )

    # ================================================================
    # 8. Validate final submission
    # ================================================================

    print("\n" + "-" * 70)
    print("VALIDATING FINAL SUBMISSION")
    print("-" * 70)

    print(
        f"  Rows: {len(submission)}"
    )

    print(
        f"  Columns: {len(submission.columns)}"
    )

    print(
        f"  Null values: "
        f"{int(submission.isna().sum().sum())}"
    )

    duplicate_count = int(
        submission.duplicated(
            [
                "loan_id",
                "reporting_month",
            ]
        ).sum()
    )

    print(
        f"  Duplicate loan-month rows: "
        f"{duplicate_count}"
    )

    print(
        f"  Output: {output_path}"
    )

    print("\n" + "=" * 70)
    print("SUBMISSION GENERATION COMPLETE")
    print("=" * 70)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()