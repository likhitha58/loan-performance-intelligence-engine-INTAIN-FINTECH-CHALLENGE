# Synthetic Data Generation — Documentation

**Provenance:** This dataset was synthetically generated for the Intain
FinTech Challenge 2026 AI Track prototype because an organizer-provided
judging data pack was not available at the time of development. It is
not official Intain data, and no statistic below represents a real
mortgage/loan portfolio.

**ORGANIZER DATA CAN REPLACE THESE FILES LATER** with minimal downstream
changes - see the schema notes in data_dictionary.md.

Generated: 2026-08-25T19:53:08.175210+00:00

## 1. How the data was generated

1. **Static attributes** are sampled per loan from documented categorical
   distributions (credit score / LTV / DTI bands, geography, purpose,
   occupancy, property type, servicer). A per-loan `composite_risk` score
   and `prepay_propensity` score are derived from these bands plus
   loan-level idiosyncratic noise; these internal scores are used only to
   drive the simulation and are not exported as columns.
2. **Monthly panels** are simulated loan-by-loan, month-by-month, as a
   Markov-style state machine over `CURRENT -> DELINQUENT -> DEFAULT` /
   `PREPAID -> CLOSED`, with transition probabilities scaled by each
   loan's risk/prepay score and the active macro scenario multipliers.
   `current_balance` evolves from the previous month via a standard
   fixed-payment amortization formula (not resampled independently). A
   CLOSED row always reports current_balance = 0.0 and days_past_due = 0,
   regardless of whether the loan arrived via PREPAID or via DEFAULT (see
   the CLOSED-state semantics note in data_dictionary.md).
3. **Targets** (`next_3m_delinquency_flag`, `next_6m_delinquency_flag`,
   `next_12m_default_flag`, `next_12m_prepayment_flag`, `next_state`) are
   computed by looking forward at each loan's own *already-simulated*
   future observations (strictly t+1 onward, never the current row t) —
   never sampled independently of the panel, and computed **before**
   anomaly injection runs.
4. **Anomalies** are injected only after the clean panel and clean
   targets are finalized, at documented per-category rates (see below),
   and are restricted to feature columns only - target columns are never
   touched (asserted at runtime; see section 5).
5. **Servicer updates** are generated as a second, partially independent
   source for a sampled subset of loan-months, with a documented fraction
   of deliberate conflicts against the core record.

## 2. Target-generation definitions

- `next_3m_delinquency_flag` = 1 if the loan is observed in DELINQUENT
  status at least once in the 3 months strictly after the current
  observation (t+1..t+3 only - t itself is never counted), else 0. Null
  if fewer than 3 future months are available AND the loan's exit from
  the observation window is not otherwise known (see horizon handling
  below).
- `next_6m_delinquency_flag` = same rule, 6-month window (t+1..t+6).
- `next_12m_default_flag` = 1 if the loan is observed in DEFAULT status
  at least once in the 12 months strictly after the current observation
  (t+1..t+12).
- `next_12m_prepayment_flag` = 1 if the loan is observed in PREPAID
  status at least once in the 12 months strictly after the current
  observation (t+1..t+12).
- `next_state` = `current_status` of the immediately following monthly
  observation for the same loan, or `CLOSED` if the loan is known to have
  reached CLOSED (even if that specific row was trimmed by the max
  observation cap), or null if the loan simply exits the generation
  window with an unknown future.

This logic is cross-checked at generation time against an independently
implemented (non-vectorized) recomputation on a random sample of loans;
the result is reported in reports/validation_report.json under
`target_logic_self_check`.

## 3. Target availability / horizon handling

Rows within `horizon_default` (12) months of a loan's last generated
observation may lack enough future months to compute all targets. These
rows have `target_horizon_available = False` internally.

**Chosen strategy:** insufficient-horizon rows are **removed from the
training file** (a supervised model must not be trained on undefined
labels). They are not removed from the raw simulated panel used
internally for statistics. This is a Prototype assumption, not a
challenge requirement — an alternative valid strategy would be to keep
the rows with explicit null target columns and let downstream code decide
whether to filter.

## 4. Train / test temporal design

- **Training period:** `reporting_month < 2024-07-01`
- **Test period:** `reporting_month >= 2024-07-01`
- **Cutoff date:** `2024-07-01` (Prototype assumption)
- The test file has all five target columns removed entirely, so no
  future information can leak into a model trained by simply reading the
  file. The training file only includes rows whose full target horizon
  falls before the panel's hard end, so training targets are always
  computed from real, already-simulated future months — never
  extrapolated or fabricated.
- Because both files are cut from the *same* underlying loan population,
  some loans appear in both train (earlier months) and test (later
  months) — this is realistic for a performance panel and is documented
  here explicitly so it is treated deliberately, not accidentally, during
  modeling (e.g. grouped cross-validation by loan_id is recommended in
  addition to the temporal cutoff).

## 5. Leakage control

- Every feature column in a given row reflects information dated at or
  before that row's `reporting_month`; no column is populated using a
  later month's values.
- Target columns are physically absent from the test file.
- Target columns are computed once, before anomaly injection, and
  anomaly injection is asserted at runtime to never modify them - if that
  invariant is ever violated the generator raises `AssertionError` and
  refuses to write output, rather than silently shipping a leaked file.
- `next_state`, all `next_*` flags, and `exception_type` are excluded
  from the static attributes and servicer_updates files, so they cannot
  leak in through a join.
- Same-loan leakage across the train/test boundary is not prevented (see
  above) but is explicitly documented so it can be addressed with
  loan-grouped validation splits.

## 6. Anomaly injection (by category, injected independently within each
of train/test at the configured rate)

| Category | Rate | What it does | exception_required set? |
|---|---|---|---|
| balance_inconsistency | 0.01 | Balance jumps up implausibly vs. prior trend | Yes |
| date_inconsistency | 0.006 | last_updated_at set before reporting_month | Yes |
| delinquency_status_inconsistency | 0.008 | days_past_due forced to 0 (creates a status/DPD mismatch on DELINQUENT rows) | Yes |
| prepayment_status_inconsistency | 0.004 | prepayment_flag=1 while current_status != PREPAID | Yes |
| default_status_inconsistency | 0.004 | default_flag=1 while current_status != DEFAULT | Yes |
| missing_document_status | 0.02 | document_status set to null | Yes |
| impossible_loan_age | 0.003 | loan_age_months inflated by +999 | Yes |
| invalid_remaining_term | 0.003 | remaining_term_months set to -1 | Yes |
| unexpected_missing_value | 0.015 | One of interest_rate / credit_score_band / servicer_name / current_balance nulled | Yes |
| static_monthly_attribute_conflict | 0.006 | Static file's credit_score_band deliberately reassigned to a *different* band than the loan's own original value | N/A (static file only - no exception_required column there) |

Every anomaly row is logged (loan_id, reporting_month, row index,
category) so it is independently auditable; see
`reports/anomaly_log.csv` after running the generator.
`exception_required` / `exception_type` on the monthly files are
populated directly from this log, not from a separate random draw, and
`exception_type` lists the exact anomaly categor(y/ies) applied
(pipe-separated if a row was touched by more than one), never a generic
label.

## 7. Source-conflict generation (servicer_updates.csv)

A configured fraction of loan-months (0.55)
receive an independent servicer update. Within that subset, a configured
fraction (0.08) is deliberately generated to
disagree with the core monthly record on status/balance/DPD
(`conflict_with_core_record = 1`), to support source-reconciliation and
conflict-detection exercises. The remaining updates are consistent with
the core record (allowing for normal reporting lag). Validation
explicitly joins servicer_updates.csv back to the core panel to confirm
every conflict-flagged row actually disagrees, and that both conflicting
and non-conflicting rows are present.

## 8. Assumption log

| Parameter | Value | Reason | Source |
|---|---|---|---|
| SEED | 42 | Reproducibility | Synthetic-generation choice |
| n_loans | 5000 | Prototype scale target | Synthetic-generation choice |
| max_observations_per_loan | 36 | Prototype scale target | Synthetic-generation choice |
| origination_start / origination_end | 2019-01-01 / 2023-06-01 | Spread origination cohorts for realistic loan-age variety | Prototype assumption |
| panel_hard_end | 2026-06-01 | Upper bound on any generated reporting_month | Prototype assumption |
| train_test_cutoff | 2024-07-01 | Chronological train/test split point | Prototype assumption |
| active_scenario_for_generation | BASE | Macro scenario used to drive the single generated dataset | Prototype assumption |
| horizon_delinquency_short/long, horizon_default, horizon_prepayment | 3 / 6 / 12 / 12 months | Matches the challenge's example target set | Challenge-specified requirement (target set), horizon lengths are a Prototype assumption |

## 9. Known limitations

- This is a simplified prototype, not a production-grade mortgage
  cash-flow / servicing engine (no escrow, no partial curtailments beyond
  the modeled prepay event, no tax/insurance modeling).
- Categorical distributions and hazard-rate base values are hand-set
  Prototype assumptions, not fitted to any real portfolio.
- Same-loan rows can appear on both sides of the train/test cutoff (see
  section 4) — address with loan-grouped validation if this matters for a
  given modeling approach.
- The macro scenario actually used to generate the one released dataset
  is `BASE`; the other two scenario
  rows in `macro_scenarios.csv` are reference parameter sets for a future
  scenario-simulation engine, not separately materialized panels.
- Designed to be replaced by an organizer-provided data pack with minimal
  changes: column names in the canonical monthly schema match the
  challenge's documented example fields.
- No PII of any kind is generated or permitted (see data_dictionary.md).

## 10. Generated dataset statistics

- Unique loans (static file): 5000
- Train rows: 104071
- Test rows: 17392
- Total monthly observations (train+test): 121463
- Train reporting_month range: 2019-01-01 to 2024-06-01
- Test reporting_month range: 2024-07-01 to 2026-05-01
- Default events (current_status==DEFAULT) train+test: 1565
- Prepayment events (current_status==PREPAID) train+test: 1621
- Delinquency observations (current_status==DELINQUENT) train+test: 3759
- Anomaly rows logged: 8889
- Servicer update rows: 80675 (conflicts: 6454)
- Validation: 26 strict checks passed, 0 failed
- Validation: 6188 expected anomaly-driven rule violations, 0 unexpected
- Target-logic self-check: 427 rows checked, 0 mismatches
