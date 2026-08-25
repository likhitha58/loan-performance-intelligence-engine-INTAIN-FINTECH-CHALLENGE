# Intain Loan Performance Intelligence Engine
## Canonical Data Schema

> Status: Prototype schema
>
> This schema is based on the example training fields and targets specified
> in the Intain FinTech Challenge 2026 AI Track problem statement.
>
> The prototype data generated from this schema is participant-generated
> synthetic data and is not represented as organizer-provided data.

---

## 1. Purpose

This document defines the canonical data contract used by the prototype
Loan Performance Intelligence Engine.

The project is designed so that the participant-generated prototype data
can later be replaced by the organizer-provided data pack without requiring
a redesign of the downstream ML pipeline.

---

## 2. Monthly Performance Fields

### Identity and Time

| Field | Description | Category |
|---|---|---|
| loan_id | Unique loan identifier | Identifier |
| month_index | Sequential monthly observation index | Temporal |
| reporting_month | Monthly performance reporting period | Temporal |
| origination_month | Loan origination month | Temporal |
| loan_age_months | Age of the loan at the reporting period | Derived temporal |
| remaining_term_months | Remaining loan term at the reporting period | Derived temporal |

### Financial

| Field | Description | Category |
|---|---|---|
| original_balance | Original loan balance | Static financial |
| current_balance | Loan balance at the reporting period | Monthly financial |
| interest_rate | Loan interest rate | Financial |

### Risk and Loan Characteristics

| Field | Description | Category |
|---|---|---|
| credit_score_band | Credit-score risk band | Risk |
| ltv_band | Loan-to-value band | Risk |
| dti_band | Debt-to-income band | Risk |
| state | Property/loan state | Geographic |
| loan_purpose | Purpose of the loan | Loan characteristic |
| occupancy_type | Occupancy classification | Property characteristic |
| property_type | Property classification | Property characteristic |

### Servicing and Performance

| Field | Description | Category |
|---|---|---|
| servicer_name | Servicing organization identifier | Servicing |
| current_status | Current loan performance state | Performance |
| days_past_due | Current delinquency measure | Performance |
| modification_flag | Indicates whether a modification is associated with the observation | Performance |
| prepayment_flag | Indicates prepayment during the current period | Performance |
| default_flag | Indicates default during the current period | Performance |
| loss_severity_band | Loss severity category | Performance |

### Provenance and Data Quality

| Field | Description | Category |
|---|---|---|
| last_updated_at | Timestamp associated with the latest record update | Provenance |
| source_system | System/source that supplied the record | Provenance |
| document_status | Status of supporting documentation/completeness | Data quality |

---

## 3. Target Variables

The following variables are future-oriented targets.

| Target | Prediction horizon |
|---|---|
| next_3m_delinquency_flag | Next 3 months |
| next_6m_delinquency_flag | Next 6 months |
| next_12m_default_flag | Next 12 months |
| next_12m_prepayment_flag | Next 12 months |
| next_state | Future loan state |
| exception_required | Exception requirement |
| exception_type | Exception classification |

Future target information must not be used as an input feature for predictions
at the observation time.

---

## 4. Temporal Principle

For an observation at reporting period `t`, prediction features must only use
information available at or before `t`.

Future observations must not be used to construct features at `t`.

Example:

    Information through month t
             |
             v
        Feature set
             |
             v
          Model
             |
             v
    Future outcome t+1 ... t+12

This rule is required to prevent temporal and target leakage.

---

## 5. Dataset Files

The prototype follows the organizer-provided data-pack interface:

- `loan_monthly_performance_train.csv`
- `loan_monthly_performance_test.csv`
- `loan_static_attributes.csv`
- `servicer_updates.csv`
- `data_dictionary.md`
- `validation_rules.json`
- `macro_scenarios.csv`
- `submission_template.csv`

The prototype files are participant-generated synthetic/reference files
until the organizer-provided data pack becomes available.

---

## 6. Data Relationships

The primary entity is `loan_id`.

The monthly performance dataset contains multiple observations per loan,
organized by `reporting_month`.

Static loan attributes are associated with loans through `loan_id`.

Servicer updates provide a secondary source for reconciliation and conflict
detection.

Macro scenarios are consumed by the scenario simulation engine.

The data dictionary and validation rules support data understanding,
validation, documentation, and grounded reviewer assistance.

---

## 7. Leakage Controls

The following information must be reviewed carefully before use as a
prediction feature:

- future target variables
- future reporting periods
- post-outcome information
- fields updated after the prediction timestamp
- derived features calculated using future observations

Every engineered feature should document its temporal availability.

---

## 8. Prototype Data Generation Principle

Synthetic data should preserve realistic longitudinal relationships rather
than assigning every field independently at random.

Examples of relationships that should be represented include:

- loan age increasing with reporting month
- remaining term decreasing with loan age
- current balance evolving over time
- delinquency states following temporal transitions
- default and prepayment affecting future loan state
- risk characteristics influencing performance probabilities
- source-system differences creating controlled reconciliation cases

The synthetic generator must document its assumptions.

---

## 9. Organizer Data Replacement

When the actual organizer-provided data pack becomes available:

1. Inspect the organizer files.
2. Compare their schema against this canonical contract.
3. Use an adapter/normalization layer where necessary.
4. Update validation rules using the organizer-provided rules.
5. Replace prototype data without redesigning the ML architecture.
6. Re-run all schema, leakage, model, and submission tests.

The organizer-provided data and documentation will take precedence over
prototype assumptions.