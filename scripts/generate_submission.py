from pathlib import Path

import pandas as pd
from src.modeling.split import temporal_split
from src.data_intelligence.features import build_features
from src.data_intelligence.loader import load_data_pack
from src.modeling.anomaly import detect_anomalies
from src.modeling.gradient_boosting import fit_gradient_boosting
from src.modeling.global_explainability import (
    build_feature_importance_table,
    build_global_explainability_summary,
)
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
from src.submission import build_submission, write_submission


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


def main():
    print("=" * 70)
    print("LOAN PERFORMANCE INTELLIGENCE ENGINE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------

    print("\nLoading data...")

    data = load_data_pack()

    train = data["train"]
    test = data["test"]

    print(f"  Train rows: {len(train)}")
    print(f"  Test rows:  {len(test)}")

    # ------------------------------------------------------------------
    # EVENT PREDICTION MODELS
    # ------------------------------------------------------------------

    print("\n" + "-" * 70)
    print("TRAINING EVENT PREDICTION MODELS")
    print("-" * 70)

    prediction_data = test[
        ["loan_id", "reporting_month"]
    ].copy()

    global_importance_rows = []

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
            threshold=0.20,
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

        # --------------------------------------------------------------
        # GLOBAL FEATURE IMPORTANCE
        # --------------------------------------------------------------

                # Use the same chronological training partition that
        # fit_gradient_boosting() used for model fitting.
        train_part, _ = temporal_split(train)

        importance_features = build_features(train_part)

        X_importance = importance_features[features].copy()
        y_importance = train_part[target].astype(int)

        importance_table = build_feature_importance_table(
            pipeline,
            features,
            X=X_importance,
            y=y_importance,
        )

        importance_summary = (
            build_global_explainability_summary(
                model_name=target,
                importance_table=importance_table,
                top_n=len(importance_table),
            )
        )

        global_importance_rows.append(
            importance_summary
        )

    # ------------------------------------------------------------------
    # SAVE GLOBAL EXPLAINABILITY REPORT
    # ------------------------------------------------------------------

    print("\n" + "-" * 70)
    print("BUILDING GLOBAL EXPLAINABILITY REPORT")
    print("-" * 70)

    global_importance = pd.concat(
        global_importance_rows,
        ignore_index=True,
    )

    global_importance_path = Path(
        "submission/global_feature_importance.csv"
    )

    global_importance.to_csv(
        global_importance_path,
        index=False,
    )

    print(
        "  Global feature importance report:",
        global_importance_path,
    )

    # ------------------------------------------------------------------
    # MACRO SCENARIO ANALYSIS
    # ------------------------------------------------------------------

    print("\n" + "-" * 70)
    print("RUNNING MACRO SCENARIO ANALYSIS")
    print("-" * 70)

    scenario_predictions = prediction_data.merge(
        test,
        on=[
            "loan_id",
            "reporting_month",
        ],
        how="left",
        suffixes=("", "_source"),
    )

    scenario_outputs = run_all_macro_scenarios(
        scenario_predictions,
        segment_columns=[
            "credit_score_band",
            "vintage",
            "servicer",
            "loan_state",
        ],
    )

    scenario_summary_rows = []

    for scenario_name, scenario_output in (
        scenario_outputs.items()
    ):
        summary = summarize_macro_scenario(
            scenario_output
        )

        scenario_summary_rows.append(summary)

    scenario_summary = pd.DataFrame(
        scenario_summary_rows
    )

    scenario_summary_path = Path(
        "submission/scenario_summary.csv"
    )

    scenario_summary.to_csv(
        scenario_summary_path,
        index=False,
    )

    segment_rows = []

    for scenario_name, scenario_output in (
        scenario_outputs.items()
    ):

        for segment_column in [
            "credit_score_band",
            "vintage",
            "servicer",
            "loan_state",
        ]:

            if segment_column not in (
                scenario_output.columns
            ):
                continue

            segment_result = (
                summarize_segment_impacts(
                    scenario_output,
                    segment_column,
                )
            )

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

        scenario_segment_impacts.to_csv(
            "submission/scenario_segment_impacts.csv",
            index=False,
        )

    print(
        "  Portfolio scenario report:",
        scenario_summary_path,
    )

    print(
        "  Segment scenario report:",
        "submission/scenario_segment_impacts.csv",
    )

    # ------------------------------------------------------------------
    # NEXT-STATE TRANSITION MODEL
    # ------------------------------------------------------------------

    print("\n" + "-" * 70)
    print("TRAINING NEXT-STATE TRANSITION MODEL")
    print("-" * 70)

    (
        transition_pipeline,
        transition_features,
        metrics,
    ) = fit_transition_model(train)

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

    transition_data[
        "next_state_confidence"
    ] = (
        transition_predictions[
            "next_state_confidence"
        ].to_numpy()
    )

    # ------------------------------------------------------------------
    # ANOMALY DETECTION
    # ------------------------------------------------------------------

    print("\n" + "-" * 70)
    print("DETECTING ANOMALIES")
    print("-" * 70)

    anomaly_data = detect_anomalies(test)

    anomaly_count = (
        anomaly_data["exception_type"] != ""
    ).sum()

    print(
        "  Anomalies detected:",
        anomaly_count,
    )

    print(
        "  Normal observations:",
        len(anomaly_data) - anomaly_count,
    )

    # ------------------------------------------------------------------
    # GROUNDED RISK EVIDENCE
    # ------------------------------------------------------------------

    print("\n" + "-" * 70)
    print("BUILDING RISK EVIDENCE")
    print("-" * 70)

    test_features = build_features(test)

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
        ] >= 0.20
    )

    risk_predictions["threshold"] = 0.20

    risk_evidence = add_risk_evidence(
        risk_predictions,
        test_features,
    )

    risk_output = (
        build_risk_intelligence_output(
            risk_evidence
        )
    )

    risk_output["reasons"] = (
        risk_output["risk_evidence"].apply(
            lambda values: (
                values
                if isinstance(values, list)
                else []
            )
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
        "  Risk evidence generated for",
        len(risk_output),
        "observations",
    )

    # ------------------------------------------------------------------
    # FINAL SUBMISSION
    # ------------------------------------------------------------------

    print("\n" + "-" * 70)
    print("BUILDING FINAL SUBMISSION")
    print("-" * 70)

    submission = build_submission(
        prediction_data,
        transition_data=transition_data,
        anomaly_data=anomaly_data,
        explanation_data=risk_output,
    )

    output_path = Path(
        "submission/submission.csv"
    )

    write_submission(
        submission,
        output_path,
    )

    # ------------------------------------------------------------------
    # FINAL VALIDATION
    # ------------------------------------------------------------------

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
        "  Null values:",
        int(submission.isna().sum().sum()),
    )

    duplicate_count = submission.duplicated(
        [
            "loan_id",
            "reporting_month",
        ]
    ).sum()

    print(
        "  Duplicate loan-month rows:",
        int(duplicate_count),
    )

    print(
        "  Output:",
        output_path,
    )

    print("\n" + "=" * 70)
    print("SUBMISSION GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()