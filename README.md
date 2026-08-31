# Loan Performance Intelligence Engine

## INTAIN Fintech Challenge — Loan Performance Intelligence & Reviewer Risk Intelligence

An end-to-end machine learning and reviewer-risk-intelligence platform for analyzing loan performance, predicting future credit events, detecting anomalous observations, understanding portfolio risk, simulating macroeconomic scenarios, and translating model outputs into actionable reviewer intelligence.

---

## 1. Executive Summary

The **Loan Performance Intelligence Engine** is a decision-support platform developed for the **INTAIN Fintech Challenge**.

The system transforms historical loan-performance data into a structured intelligence layer that helps portfolio reviewers answer:

- Which loans are likely to deteriorate?
- Which loans require attention?
- What evidence is driving the predicted risk?
- What state is a loan likely to transition into?
- Which loan observations appear anomalous?
- Which portfolio segments are exposed to adverse scenarios?
- How should reviewers prioritize their workload?
- How can model predictions be converted into understandable explanations and recommended actions?

Rather than treating machine-learning predictions as standalone outputs, the solution connects:

**Data → Features → Predictions → Anomalies → Evidence → Risk → Actions → Explanations → Portfolio Intelligence → Reviewer Experience**

The final system includes:

- Data intelligence and quality checks
- Temporal and behavioral feature engineering
- Multi-horizon delinquency prediction
- 12-month default prediction
- 12-month prepayment prediction
- Next-state transition prediction
- Probability calibration and thresholding components
- Isolation Forest anomaly detection
- Risk evidence generation
- Risk-tier classification
- Reviewer action prioritization
- Loan-level explanations
- Global feature importance
- Macro scenario simulation
- Portfolio and segment intelligence
- Evidence-grounded LLM reviewer assistance
- Streamlit reviewer dashboard
- Automated submission generation
- Automated validation
- Comprehensive automated testing

The system is designed as a **reviewer decision-support system**, not as an autonomous lending or credit-approval engine.

---

# 2. Challenge Context

The challenge focuses on extracting actionable intelligence from loan-performance information.

A useful solution must go beyond simply predicting whether an event will occur. A reviewer needs to understand:

1. **What is likely to happen?**
2. **When might it happen?**
3. **Why is the observation considered risky?**
4. **How severe is the risk?**
5. **What should the reviewer do next?**
6. **Is the observation unusual compared with the broader portfolio?**
7. **How could changing macroeconomic conditions affect the portfolio?**

The Loan Performance Intelligence Engine addresses these requirements through a layered architecture combining predictive modeling, anomaly detection, explainability, scenario analysis, and reviewer-oriented decision support.

---

# 3. Solution Overview

The solution follows the following high-level workflow:

```text
                     RAW LOAN DATA
                           │
                           ▼
              ┌─────────────────────────┐
              │   DATA INTELLIGENCE     │
              │                         │
              │ • Profiling             │
              │ • Missingness            │
              │ • Outliers               │
              │ • Record Quality         │
              │ • Relationship Checks    │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  FEATURE ENGINEERING    │
              │                         │
              │ • Historical Behavior    │
              │ • Delinquency Signals   │
              │ • Status Changes        │
              │ • Credit / LTV / DTI    │
              │ • Modification Signals  │
              └────────────┬────────────┘
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
       EVENT MODELS    STATE MODEL    ANOMALY MODEL
             │             │             │
             │             │             │
       ┌─────┴─────┐       │       Isolation Forest
       │           │       │             │
       ▼           ▼       ▼             ▼
      3M          6M    Next State     Anomaly
   Delinquency Delinq. Prediction      Score
       │           │       │             │
       └───────────┼───────┼─────────────┘
                   │
                   ▼
          ┌───────────────────────┐
          │   RISK INTELLIGENCE   │
          │                       │
          │ • Risk Tier           │
          │ • Evidence Category   │
          │ • Action Priority     │
          │ • Confidence          │
          └───────────┬───────────┘
                      │
        ┌─────────────┼──────────────┐
        │             │              │
        ▼             ▼              ▼
 EXPLAINABILITY   SCENARIO       PORTFOLIO
                  ANALYSIS       INTELLIGENCE
        │             │              │
        └─────────────┼──────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │ REVIEWER EXPERIENCE   │
          │                       │
          │ • Streamlit Dashboard │
          │ • Reviewer Queue      │
          │ • Loan Explanations   │
          │ • Portfolio Insights  │
          └───────────┬───────────┘
                      │
                      ▼
             FINAL SUBMISSION
          submission/submission.csv


