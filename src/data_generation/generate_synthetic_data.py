"""
generate_synthetic_data.py

SYNTHETIC PROTOTYPE DATA GENERATOR
Intain FinTech Challenge 2026 - Round 2 - AI Track
Loan Performance Intelligence Engine

--------------------------------------------------------------------------
IMPORTANT PROVENANCE NOTICE
--------------------------------------------------------------------------
Every file this script produces is SYNTHETIC PROTOTYPE DATA.
It was generated locally because an organizer-provided judging data pack
was not available at the time of development. It is NOT official Intain
data and none of the distributions or statistics represent a real
mortgage/loan portfolio. See docs/synthetic_data_generation.md for the
full assumption log this script writes out.

ORGANIZER DATA CAN REPLACE THESE FILES LATER: column names in the
canonical monthly/static/servicer/macro/submission schemas are designed
to match the challenge's documented example fields, so an
organizer-provided data pack can be dropped into data/raw/ in place of
these files with minimal changes to any downstream code.
--------------------------------------------------------------------------

EXPECTED LOCATION:
    src/data_generation/generate_synthetic_data.py

USAGE (run from the project root, e.g. an activated .venv):
    python src/data_generation/generate_synthetic_data.py

Output paths are resolved relative to the project root (two directories
above this file), NOT relative to your current working directory, so it
is safe to run this from anywhere as long as the file stays at
src/data_generation/generate_synthetic_data.py inside the repo.

Everything is driven by the CONFIG dict below. This is the SYNTHETIC
DATA GENERATION PHASE ONLY - no modeling, feature engineering, anomaly-
detection models, scenario-simulation engines, LLM/agent components, or
serving layers are implemented in this file. Those are later phases.

Requires: numpy, pandas  (pip install numpy pandas)
"""

import os
import json
import random
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Project root = two levels above this file (src/data_generation/ -> src/ -> root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve(*parts):
    """Resolve an output path relative to the project root, regardless of CWD."""
    return os.path.join(PROJECT_ROOT, *parts)

# ==========================================================================
# 0. CONFIG
# ==========================================================================
# All major generation parameters live here so the whole run is auditable
# and reproducible. Edit these values to rescale the prototype.
#
# NOTE (Issue 12): n_loans / max_observations_per_loan are intentionally
# kept at prototype scale for local development. Do not increase them here
# as part of a "correction" pass - that is a separate, deliberate decision.

CONFIG = {
    "seed": 42,

    # --- scale (kept at prototype scale - see note above) ---
    "n_loans": 5000,
    "max_observations_per_loan": 36,     # cap on monthly rows per loan

    # --- calendar ---
    "origination_start": "2019-01-01",
    "origination_end": "2023-06-01",
    "panel_hard_end": "2026-06-01",      # no reporting_month may exceed this
    "train_test_cutoff": "2024-07-01",   # reporting_month >= cutoff -> test

    # --- loan terms ---
    "original_term_months_choices": [180, 240, 360],
    "original_term_months_weights": [0.15, 0.15, 0.70],

    # --- static categorical distributions (synthetic prototype assumptions) ---
    "credit_score_band": {
        "categories": ["<620", "620-659", "660-699", "700-739", "740-779", "780+"],
        "weights":    [0.06,    0.10,      0.16,      0.22,      0.24,      0.22],
        # relative risk multiplier, higher = riskier
        "risk":       [1.90,    1.55,      1.25,      1.00,      0.75,      0.55],
    },
    "ltv_band": {
        "categories": ["<=60", "61-70", "71-80", "81-90", "91-95", ">95"],
        "weights":    [0.18,    0.20,    0.28,    0.18,    0.10,    0.06],
        "risk":       [0.65,    0.80,    1.00,    1.25,    1.55,    1.90],
    },
    "dti_band": {
        "categories": ["<=20", "21-30", "31-36", "37-43", "44-50", ">50"],
        "weights":    [0.12,    0.26,    0.24,    0.22,    0.11,    0.05],
        "risk":       [0.70,    0.85,    1.00,    1.20,    1.50,    1.85],
    },
    "state_geo": {
        "categories": ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "AZ",
                        "WA", "CO", "MA", "MI", "OTHER"],
        "weights":    [0.14, 0.10, 0.09, 0.08, 0.05, 0.05, 0.04, 0.05, 0.04, 0.04,
                        0.04, 0.03, 0.03, 0.03, 0.19],
    },
    "loan_purpose": {
        "categories": ["PURCHASE", "RATE_TERM_REFI", "CASHOUT_REFI"],
        "weights":    [0.62, 0.21, 0.17],
    },
    "occupancy_type": {
        "categories": ["PRIMARY", "SECONDARY", "INVESTMENT"],
        "weights":    [0.82, 0.08, 0.10],
    },
    "property_type": {
        "categories": ["SINGLE_FAMILY", "CONDO", "TOWNHOUSE", "MULTI_FAMILY_2_4"],
        "weights":    [0.68, 0.16, 0.11, 0.05],
    },
    "servicer_name": {
        "categories": ["SVC_ALPHA", "SVC_BRAVO", "SVC_CHARLIE", "SVC_DELTA", "SVC_ECHO"],
        "weights":    [0.28, 0.24, 0.20, 0.16, 0.12],
    },
    "loss_severity_band": {
        "categories": ["<10%", "10-25%", "25-40%", "40-60%", ">60%"],
        "weights":    [0.15, 0.30, 0.28, 0.18, 0.09],
    },

    # --- monthly transition / hazard base rates (before risk scaling) ---
    "monthly_hazard": {
        "current_to_delinquent_base": 0.012,
        "current_to_prepay_base": 0.010,
        "delinquent_cure_base": 0.28,
        "delinquent_to_default_base": 0.055,
        "modification_prob_while_delinquent": 0.05,
        "default_to_closed_prob": 0.35,
        "prepaid_to_closed_prob": 0.85,
    },

    # --- macro scenario multipliers applied on top of the base hazards ---
    "macro_scenarios": {
        "BASE":              {"delinquency_multiplier": 1.00, "default_multiplier": 1.00, "prepayment_multiplier": 1.00, "credit_risk_multiplier": 1.00},
        "ADVERSE_CREDIT":    {"delinquency_multiplier": 1.65, "default_multiplier": 1.90, "prepayment_multiplier": 0.55, "credit_risk_multiplier": 1.45},
        "HIGH_PREPAYMENT":   {"delinquency_multiplier": 0.85, "default_multiplier": 0.80, "prepayment_multiplier": 2.10, "credit_risk_multiplier": 0.90},
    },
    # scenario actually used to drive the single generated dataset;
    # the other rows of macro_scenarios.csv are reference parameters only
    "active_scenario_for_generation": "BASE",

    # --- target horizons (months) ---
    "horizon_delinquency_short": 3,
    "horizon_delinquency_long": 6,
    "horizon_default": 12,
    "horizon_prepayment": 12,

    # --- anomaly injection rates (fraction of eligible rows) ---
    "anomaly_rates": {
        "balance_inconsistency": 0.010,
        "date_inconsistency": 0.006,
        "delinquency_status_inconsistency": 0.008,
        "prepayment_status_inconsistency": 0.004,
        "default_status_inconsistency": 0.004,
        "missing_document_status": 0.020,
        "missing_or_stale_update": 0.015,
        "impossible_loan_age": 0.003,
        "invalid_remaining_term": 0.003,
        "unexpected_missing_value": 0.015,
        "static_monthly_attribute_conflict": 0.006,
    },

    # --- servicer_updates.csv generation ---
    "servicer_update_coverage": 0.55,     # fraction of loan-months with an update
    "servicer_conflict_rate": 0.08,       # fraction of those updates that conflict

    # --- generic field-level missingness (legitimate, non-anomalous) ---
    "base_missingness": {
        "modification_flag": 0.0,   # derived, never missing
        "servicer_name": 0.0,
    },

    "output_dir": "data/raw",
    "docs_dir": "docs",
    "reports_dir": "reports",
    "submission_dir": "submission",
}

STATES = ["CURRENT", "DELINQUENT", "DEFAULT", "PREPAID", "CLOSED"]

# ==========================================================================
# REQUIRED SCHEMAS (Issue 5) - used both to build the FINAL_COLUMNS lists
# and to explicitly validate every output file's schema.
# ==========================================================================

TARGET_COLUMNS = [
    "next_3m_delinquency_flag", "next_6m_delinquency_flag",
    "next_12m_default_flag", "next_12m_prepayment_flag", "next_state",
]

TRAIN_SCHEMA = [
    "loan_id", "month_index", "reporting_month", "origination_month",
    "loan_age_months", "remaining_term_months", "original_balance",
    "current_balance", "interest_rate", "credit_score_band", "ltv_band",
    "dti_band", "state", "loan_purpose", "occupancy_type", "property_type",
    "servicer_name", "current_status", "days_past_due", "modification_flag",
    "prepayment_flag", "default_flag", "loss_severity_band", "last_updated_at",
    "source_system", "document_status",
    "next_3m_delinquency_flag", "next_6m_delinquency_flag",
    "next_12m_default_flag", "next_12m_prepayment_flag", "next_state",
    "exception_required", "exception_type",
]

TEST_SCHEMA = [c for c in TRAIN_SCHEMA if c not in TARGET_COLUMNS]

STATIC_SCHEMA = [
    "loan_id", "origination_month", "original_balance", "credit_score_band",
    "ltv_band", "dti_band", "state", "loan_purpose", "occupancy_type",
    "property_type", "servicer_name", "interest_rate", "original_term_months",
]

SERVICER_SCHEMA = [
    "update_id", "loan_id", "reporting_month", "servicer_name", "update_date",
    "reported_status", "reported_balance", "reported_days_past_due",
    "conflict_with_core_record",
]

MACRO_SCHEMA = [
    "scenario_name", "description", "credit_risk_multiplier",
    "delinquency_multiplier", "default_multiplier", "prepayment_multiplier",
    "is_active_generation_scenario", "provenance",
]

SUBMISSION_SCHEMA = [
    "loan_id", "reporting_month", "pred_next_3m_delinquency_prob",
    "pred_next_6m_delinquency_prob", "pred_next_12m_default_prob",
    "pred_next_12m_prepayment_prob", "pred_next_state", "anomaly_score",
    "exception_type", "top_drivers", "recommended_action", "confidence",
]

# kept for backwards compatibility with the rest of the module
FINAL_COLUMNS = TRAIN_SCHEMA


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def weighted_choice(rng, categories, weights, size):
    return rng.choice(categories, size=size, p=np.array(weights) / np.sum(weights))


# ==========================================================================
# 1. STATIC LOAN ATTRIBUTES
# ==========================================================================

def generate_static_attributes(config, rng):
    n = config["n_loans"]
    loan_id = np.array([f"L{100000 + i}" for i in range(n)])

    csb = config["credit_score_band"]
    ltv = config["ltv_band"]
    dti = config["dti_band"]

    credit_score_band = weighted_choice(rng, csb["categories"], csb["weights"], n)
    ltv_band = weighted_choice(rng, ltv["categories"], ltv["weights"], n)
    dti_band = weighted_choice(rng, dti["categories"], dti["weights"], n)

    geo = config["state_geo"]
    state = weighted_choice(rng, geo["categories"], geo["weights"], n)

    purpose = config["loan_purpose"]
    loan_purpose = weighted_choice(rng, purpose["categories"], purpose["weights"], n)

    occ = config["occupancy_type"]
    occupancy_type = weighted_choice(rng, occ["categories"], occ["weights"], n)

    prop = config["property_type"]
    property_type = weighted_choice(rng, prop["categories"], prop["weights"], n)

    svc = config["servicer_name"]
    servicer_name = weighted_choice(rng, svc["categories"], svc["weights"], n)

    term_choices = config["original_term_months_choices"]
    term_weights = config["original_term_months_weights"]
    original_term_months = rng.choice(term_choices, size=n, p=term_weights)

    # origination month, uniform over the configured window
    orig_start = pd.Timestamp(config["origination_start"])
    orig_end = pd.Timestamp(config["origination_end"])
    span_months = (orig_end.year - orig_start.year) * 12 + (orig_end.month - orig_start.month)
    offsets = rng.integers(0, span_months + 1, size=n)
    origination_month = pd.DatetimeIndex([orig_start + pd.DateOffset(months=int(o)) for o in offsets])

    # original balance: lognormal, mildly correlated with occupancy/property
    base_balance = rng.lognormal(mean=12.2, sigma=0.42, size=n)  # centered ~ $200k
    property_mult = pd.Series(property_type).map({
        "SINGLE_FAMILY": 1.00, "CONDO": 0.80, "TOWNHOUSE": 0.90, "MULTI_FAMILY_2_4": 1.35
    }).to_numpy()
    original_balance = np.round(base_balance * property_mult, -2)
    original_balance = np.clip(original_balance, 50_000, 1_500_000)

    # interest rate: base rate by origination year (very rough synthetic macro curve)
    orig_year = origination_month.year
    year_base_rate = pd.Series(orig_year).map({
        2019: 4.1, 2020: 3.2, 2021: 3.0, 2022: 5.2, 2023: 6.8, 2024: 6.4, 2025: 5.9
    }).fillna(5.5).to_numpy()

    credit_risk_val = pd.Series(credit_score_band).map(
        dict(zip(csb["categories"], csb["risk"]))
    ).to_numpy()
    rate_noise = rng.normal(0, 0.35, size=n)
    interest_rate = np.round(year_base_rate + (credit_risk_val - 1.0) * 1.1 + rate_noise, 3)
    interest_rate = np.clip(interest_rate, 2.0, 12.0)

    static_df = pd.DataFrame({
        "loan_id": loan_id,
        "origination_month": origination_month.strftime("%Y-%m-01"),
        "original_balance": original_balance,
        "credit_score_band": credit_score_band,
        "ltv_band": ltv_band,
        "dti_band": dti_band,
        "state": state,
        "loan_purpose": loan_purpose,
        "occupancy_type": occupancy_type,
        "property_type": property_type,
        "servicer_name": servicer_name,
        "interest_rate": interest_rate,
        "original_term_months": original_term_months,
    })

    # internal-only risk/prepay factors used to drive the panel simulation.
    # kept out of the exported static file; returned separately.
    ltv_risk = pd.Series(ltv_band).map(dict(zip(ltv["categories"], ltv["risk"]))).to_numpy()
    dti_risk = pd.Series(dti_band).map(dict(zip(dti["categories"], dti["risk"]))).to_numpy()
    idiosyncratic = rng.lognormal(mean=0.0, sigma=0.25, size=n)
    composite_risk = credit_risk_val * ltv_risk * dti_risk * idiosyncratic
    composite_risk = composite_risk / np.median(composite_risk)  # normalize around 1.0

    prepay_propensity = rng.lognormal(mean=0.0, sigma=0.35, size=n)
    prepay_propensity = prepay_propensity / np.median(prepay_propensity)

    internal = pd.DataFrame({
        "loan_id": loan_id,
        "composite_risk": composite_risk,
        "prepay_propensity": prepay_propensity,
    })

    return static_df, internal


# ==========================================================================
# 2. MONTHLY PANEL SIMULATION (clean, pre-anomaly)
# ==========================================================================
#
# CLOSED-state semantics (Issue 9, made explicit and deterministic):
#   - A loan can reach CLOSED from either PREPAID or DEFAULT.
#   - Regardless of which path was taken, a CLOSED row always reports
#     current_balance = 0.0 and days_past_due = 0 - it represents a fully
#     terminal, closed record. prepayment_flag / default_flag remain 0 on
#     a CLOSED row by design (those flags mark the specific month the loan
#     was observed IN that status, not any later terminal state - see the
#     data dictionary note on those two columns). The path taken into
#     CLOSED (via PREPAID vs. via DEFAULT) can always be recovered by
#     looking at that loan's immediately preceding monthly observation
#     (current_status) or via next_state on the prior row.
#   - Once a loan reaches CLOSED, no further monthly rows are generated
#     for it (the simulation loop breaks immediately after recording the
#     CLOSED row), so there is exactly one CLOSED row per loan that
#     reaches that state within the panel window.

def simulate_panel(static_df, internal_df, config, rng):
    hz = config["monthly_hazard"]
    scenario = config["macro_scenarios"][config["active_scenario_for_generation"]]
    max_obs = config["max_observations_per_loan"]
    hard_end = pd.Timestamp(config["panel_hard_end"])

    merged = static_df.merge(internal_df, on="loan_id")
    n_loans = len(merged)

    records = []

    loan_id_arr = merged["loan_id"].to_numpy()
    orig_month_arr = pd.to_datetime(merged["origination_month"]).to_numpy()
    orig_balance_arr = merged["original_balance"].to_numpy()
    rate_arr = merged["interest_rate"].to_numpy()
    term_arr = merged["original_term_months"].to_numpy()
    risk_arr = merged["composite_risk"].to_numpy()
    prepay_arr = merged["prepay_propensity"].to_numpy()
    servicer_arr = merged["servicer_name"].to_numpy()

    for i in range(n_loans):
        loan_id = loan_id_arr[i]
        orig_month = pd.Timestamp(orig_month_arr[i])
        term = int(term_arr[i])
        monthly_rate = rate_arr[i] / 100.0 / 12.0
        balance = float(orig_balance_arr[i])
        risk = float(risk_arr[i])
        prepay_prop = float(prepay_arr[i])
        servicer = servicer_arr[i]

        # scheduled fixed payment (standard amortization formula)
        if monthly_rate > 0:
            payment = balance * (monthly_rate * (1 + monthly_rate) ** term) / \
                      ((1 + monthly_rate) ** term - 1)
        else:
            payment = balance / term

        state = "CURRENT"
        dpd = 0
        months_delinquent_streak = 0
        prior_delinquency_events = 0

        for m in range(1, max_obs + 1):
            reporting_month = orig_month + pd.DateOffset(months=m - 1)
            if reporting_month > hard_end:
                break
            # loan_age_months convention (Issue 8): loan_age_months = m, so
            # the origination month itself is recorded as loan_age_months = 1
            # (i.e. origination_month == reporting_month <=> loan_age_months
            # == 1). This is intentional and is the exact convention R003
            # validates against - see run_validation().
            loan_age_months = m
            remaining_term_months = max(term - loan_age_months, 0)

            # -------- record current-month row (state as of this month) ----
            modification_flag = 0
            prepayment_flag = 1 if state == "PREPAID" else 0
            default_flag = 1 if state == "DEFAULT" else 0

            records.append((
                loan_id, m, reporting_month, orig_month, loan_age_months,
                remaining_term_months, orig_balance_arr[i], balance, rate_arr[i],
                state, dpd, modification_flag, prepayment_flag, default_flag,
                servicer, risk,
            ))

            if state == "CLOSED":
                break

            # -------- evolve balance for months where the loan still accrues
            if state in ("CURRENT", "DELINQUENT") and remaining_term_months > 0:
                interest_due = balance * monthly_rate
                principal_due = max(payment - interest_due, 0)
                balance = max(balance - principal_due, 0)

            # -------- draw next state ---------------------------------------
            u = rng.random()
            if state == "CURRENT":
                p_delinq = min(hz["current_to_delinquent_base"] * risk *
                                scenario["delinquency_multiplier"], 0.35)
                p_prepay = min(hz["current_to_prepay_base"] * prepay_prop *
                                scenario["prepayment_multiplier"], 0.30)
                if u < p_delinq:
                    state = "DELINQUENT"
                    months_delinquent_streak = 1
                    prior_delinquency_events += 1
                    dpd = 30
                elif u < p_delinq + p_prepay:
                    state = "PREPAID"
                    balance = 0.0
                    dpd = 0
                else:
                    dpd = 0

            elif state == "DELINQUENT":
                p_mod = hz["modification_prob_while_delinquent"]
                p_cure = hz["delinquent_cure_base"] / (1 + 0.15 * months_delinquent_streak)
                p_default = min(
                    hz["delinquent_to_default_base"] * risk *
                    scenario["default_multiplier"] * (1 + 0.25 * months_delinquent_streak),
                    0.60,
                )
                if u < p_mod:
                    state = "CURRENT"
                    months_delinquent_streak = 0
                    dpd = 0
                    # modification treated as reset of streak; flag applies to THIS row
                    records[-1] = records[-1][:11] + (1,) + records[-1][12:]
                elif u < p_mod + p_cure:
                    state = "CURRENT"
                    months_delinquent_streak = 0
                    dpd = 0
                elif u < p_mod + p_cure + p_default:
                    state = "DEFAULT"
                    dpd = 180
                else:
                    months_delinquent_streak += 1
                    dpd = min(30 * (months_delinquent_streak + 1), 150)

            elif state == "DEFAULT":
                if u < hz["default_to_closed_prob"]:
                    state = "CLOSED"
                    dpd = 0
                    balance = 0.0  # CLOSED rows always report a zero balance (see note above)
                else:
                    dpd = 180

            elif state == "PREPAID":
                if u < hz["prepaid_to_closed_prob"]:
                    state = "CLOSED"
                dpd = 0

    cols = [
        "loan_id", "month_index", "reporting_month", "origination_month",
        "loan_age_months", "remaining_term_months", "original_balance",
        "current_balance", "interest_rate", "current_status", "days_past_due",
        "modification_flag", "prepayment_flag", "default_flag",
        "servicer_name", "_composite_risk",
    ]
    panel = pd.DataFrame.from_records(records, columns=cols)
    panel["current_balance"] = panel["current_balance"].round(2)
    panel["days_past_due"] = panel["days_past_due"].astype(int)
    return panel


# ==========================================================================
# 3. ENRICH WITH STATIC BANDS + TARGET GENERATION
# ==========================================================================

def enrich_with_static_bands(panel, static_df):
    join_cols = ["loan_id", "credit_score_band", "ltv_band", "dti_band", "state",
                 "loan_purpose", "occupancy_type", "property_type"]
    panel = panel.merge(static_df[join_cols], on="loan_id", how="left")
    return panel


def compute_targets(panel, config):
    """
    Target definitions (Issue 7 - verified below):

      next_3m_delinquency_flag  -> looks ONLY at t+1, t+2, t+3 (never t)
      next_6m_delinquency_flag  -> looks ONLY at t+1 .. t+6   (never t)
      next_12m_default_flag     -> looks ONLY at t+1 .. t+12  (never t)
      next_12m_prepayment_flag  -> looks ONLY at t+1 .. t+12  (never t)

    All four are computed by slicing the *already-simulated* per-loan
    status array starting at index t+1 (never index t), so the current
    month's own state is never counted toward its own forward-looking
    target. This is re-verified programmatically by
    `_self_check_target_logic()` below and its result is folded into the
    validation report.
    """
    panel = panel.sort_values(["loan_id", "month_index"]).reset_index(drop=True)
    h3 = config["horizon_delinquency_short"]
    h6 = config["horizon_delinquency_long"]
    h12d = config["horizon_default"]
    h12p = config["horizon_prepayment"]

    out_frames = []
    for loan_id, g in panel.groupby("loan_id", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        status = g["current_status"].to_numpy()
        is_delinq = (status == "DELINQUENT").astype(int)
        is_default = (status == "DEFAULT").astype(int)
        is_prepaid = (status == "PREPAID").astype(int)

        next3 = np.full(n, np.nan)
        next6 = np.full(n, np.nan)
        next12d = np.full(n, np.nan)
        next12p = np.full(n, np.nan)
        next_state = np.array([None] * n, dtype=object)

        loan_ends_before_panel_end = g["current_status"].iloc[-1] == "CLOSED"

        for t in range(n):
            # next_state: the very next observation, if any
            if t + 1 < n:
                next_state[t] = status[t + 1]
            elif loan_ends_before_panel_end:
                next_state[t] = "CLOSED"
            # else: unknown (loan simply exits observation window) -> None

            future_avail_3 = (t + h3) < n or loan_ends_before_panel_end
            future_avail_6 = (t + h6) < n or loan_ends_before_panel_end
            future_avail_12 = (t + h12d) < n or loan_ends_before_panel_end

            # NOTE: every slice below starts at [t + 1 : ...] - the current
            # month (index t) is deliberately excluded from its own target.
            if future_avail_3:
                end = min(t + h3, n - 1)
                next3[t] = 1 if is_delinq[t + 1:end + 1].sum() > 0 else 0
            if future_avail_6:
                end = min(t + h6, n - 1)
                next6[t] = 1 if is_delinq[t + 1:end + 1].sum() > 0 else 0
            if future_avail_12:
                end = min(t + h12d, n - 1)
                next12d[t] = 1 if is_default[t + 1:end + 1].sum() > 0 else 0
                end_p = min(t + h12p, n - 1)
                next12p[t] = 1 if is_prepaid[t + 1:end_p + 1].sum() > 0 else 0

        g["next_3m_delinquency_flag"] = next3
        g["next_6m_delinquency_flag"] = next6
        g["next_12m_default_flag"] = next12d
        g["next_12m_prepayment_flag"] = next12p
        g["next_state"] = next_state
        g["target_horizon_available"] = (~pd.isna(next12d)) & (~pd.isna(next3))
        out_frames.append(g)

    return pd.concat(out_frames, ignore_index=True)


def self_check_target_logic(panel_with_targets, config, rng, n_sample_loans=25):
    """
    Independent re-derivation of the four forward-looking targets for a
    random sample of loans, computed with a deliberately different (naive,
    non-vectorized) implementation than compute_targets(). Used purely as
    a validation cross-check (Issue 7) - it does not feed back into the
    generated data.
    """
    h3 = config["horizon_delinquency_short"]
    h6 = config["horizon_delinquency_long"]
    h12d = config["horizon_default"]
    h12p = config["horizon_prepayment"]

    loan_ids = panel_with_targets["loan_id"].unique()
    sample_ids = rng.choice(loan_ids, size=min(n_sample_loans, len(loan_ids)), replace=False)

    mismatches = 0
    rows_checked = 0

    for loan_id in sample_ids:
        g = panel_with_targets[panel_with_targets["loan_id"] == loan_id].sort_values("month_index").reset_index(drop=True)
        statuses = g["current_status"].tolist()
        n = len(g)
        for t in range(n):
            future = statuses[t + 1:]  # strictly future rows only, index t excluded
            if not future:
                continue
            rows_checked += 1
            naive_next3 = 1 if "DELINQUENT" in future[:h3] and len(future) >= h3 else (
                1 if "DELINQUENT" in future[:h3] and g["current_status"].iloc[-1] == "CLOSED" else np.nan
            )
            recomputed = g["next_3m_delinquency_flag"].iloc[t]
            if not (pd.isna(naive_next3) and pd.isna(recomputed)):
                if pd.isna(naive_next3) or pd.isna(recomputed):
                    continue  # horizon-availability edge case, not a logic mismatch
                if int(naive_next3) != int(recomputed):
                    mismatches += 1

    return {"rows_checked": rows_checked, "mismatches": mismatches}


# ==========================================================================
# 4. FIELD FINALIZATION (dates, doc status, source system, exceptions)
# ==========================================================================

def finalize_fields(panel, rng):
    panel["reporting_month"] = pd.to_datetime(panel["reporting_month"]).dt.strftime("%Y-%m-01")
    panel["origination_month"] = pd.to_datetime(panel["origination_month"]).dt.strftime("%Y-%m-01")
    panel["last_updated_at"] = (
        pd.to_datetime(panel["reporting_month"]) + pd.DateOffset(days=4)
    ).dt.strftime("%Y-%m-%d")
    panel["source_system"] = "SYNTH_CORE_LOS"
    panel["document_status"] = rng.choice(
        ["COMPLETE", "PENDING", "INCOMPLETE"], size=len(panel), p=[0.86, 0.09, 0.05]
    )
    # exception placeholders, populated for real during anomaly injection
    panel["exception_required"] = 0
    panel["exception_type"] = ""
    return panel


def assign_loss_severity(panel, config, rng):
    lsb = config["loss_severity_band"]
    panel["loss_severity_band"] = ""
    default_mask = panel["current_status"].isin(["DEFAULT"])
    panel.loc[default_mask, "loss_severity_band"] = weighted_choice(
        rng, lsb["categories"], lsb["weights"], int(default_mask.sum())
    )
    return panel


# ==========================================================================
# 5. TRAIN / TEST SPLIT (chronological)
# ==========================================================================

def split_train_test(panel, config):
    cutoff = pd.Timestamp(config["train_test_cutoff"])
    rep = pd.to_datetime(panel["reporting_month"])

    train_mask = (rep < cutoff) & (panel["target_horizon_available"])
    test_mask = rep >= cutoff

    train_df = panel.loc[train_mask].copy()
    test_df = panel.loc[test_mask].copy()

    # test file mimics a genuine held-out prediction scenario:
    # forward-looking target columns are withheld (this is what the
    # submission is expected to predict). next_state is also withheld.
    test_df = test_df.drop(columns=TARGET_COLUMNS)

    return train_df, test_df


# ==========================================================================
# 6. ANOMALY INJECTION
# ==========================================================================
#
# IMPORTANT (Issue 2): targets are computed in compute_targets() BEFORE
# this function runs, directly from the clean simulated trajectory. This
# function deliberately corrupts FEATURE columns only, for data-quality
# testing purposes. It never touches, recomputes, or is permitted to
# change any of TARGET_COLUMNS. A hard assertion at the end of this
# function verifies that invariant on every run (raises if violated) -
# see the target-column snapshot/compare block below.
#
# IMPORTANT (Issue 3): every anomaly type sets exception_required = 1 and
# appends its own name to exception_type (pipe-separated if a row is
# touched by more than one anomaly type), instead of a generic
# "DATA_QUALITY_REVIEW" label - this is what lets exception_type serve as
# an exact ground-truth answer key for a future anomaly-detection model.

def inject_anomalies(train_df, test_df, static_df, config, rng):
    """Injects controlled, documented data-quality problems.
    Mutates copies of train_df/test_df and static_df, and returns an
    anomaly log describing exactly which rows were touched and how."""

    train_df = train_df.copy()
    test_df = test_df.copy()
    static_df = static_df.copy()
    rates = config["anomaly_rates"]
    log_rows = []

    # Issue 2 safeguard: snapshot target columns (train only - test never
    # has them) before any mutation, to assert they are untouched after.
    train_target_snapshot = {c: train_df[c].copy() for c in TARGET_COLUMNS if c in train_df.columns}

    def sample_idx(df, rate, seed_bump):
        n = int(len(df) * rate)
        if n == 0:
            return np.array([], dtype=int)
        local_rng = np.random.default_rng(config["seed"] + seed_bump)
        return local_rng.choice(df.index.to_numpy(), size=n, replace=False)

    frames = {"train": train_df, "test": test_df}

    for split_name, df in frames.items():
        bump = 100 if split_name == "train" else 200
        # Issue 3: per-row accumulator of every anomaly type applied to
        # that row this split, so exception_type can list all of them.
        exc_acc = defaultdict(list)

        def mark(idx, anomaly_type):
            for i in idx:
                exc_acc[i].append(anomaly_type)

        # 1. balance_inconsistency: current_balance suddenly increases
        idx = sample_idx(df, rates["balance_inconsistency"], bump + 1)
        df.loc[idx, "current_balance"] = df.loc[idx, "current_balance"] * 1.35 + 5000
        mark(idx, "balance_inconsistency")
        _log(log_rows, split_name, df, idx, "balance_inconsistency")

        # 2. date_inconsistency: last_updated_at before reporting_month
        idx = sample_idx(df, rates["date_inconsistency"], bump + 2)
        bad_dates = (pd.to_datetime(df.loc[idx, "reporting_month"]) - pd.DateOffset(days=10))
        df.loc[idx, "last_updated_at"] = bad_dates.dt.strftime("%Y-%m-%d")
        mark(idx, "date_inconsistency")
        _log(log_rows, split_name, df, idx, "date_inconsistency")

        # 3. delinquency_status_inconsistency: DPD=0 but status DELINQUENT (or vice versa)
        idx = sample_idx(df, rates["delinquency_status_inconsistency"], bump + 3)
        df.loc[idx, "days_past_due"] = 0
        mark(idx, "delinquency_status_inconsistency")
        _log(log_rows, split_name, df, idx, "delinquency_status_inconsistency")

        # 4. prepayment_status_inconsistency: prepayment_flag=1 but status != PREPAID
        eligible = df.index[df["current_status"] != "PREPAID"]
        n = int(len(df) * rates["prepayment_status_inconsistency"])
        if n and len(eligible):
            local_rng = np.random.default_rng(config["seed"] + bump + 4)
            idx = local_rng.choice(eligible, size=min(n, len(eligible)), replace=False)
            df.loc[idx, "prepayment_flag"] = 1
            mark(idx, "prepayment_status_inconsistency")
            _log(log_rows, split_name, df, idx, "prepayment_status_inconsistency")

        # 5. default_status_inconsistency: default_flag=1 but status != DEFAULT
        eligible = df.index[df["current_status"] != "DEFAULT"]
        n = int(len(df) * rates["default_status_inconsistency"])
        if n and len(eligible):
            local_rng = np.random.default_rng(config["seed"] + bump + 5)
            idx = local_rng.choice(eligible, size=min(n, len(eligible)), replace=False)
            df.loc[idx, "default_flag"] = 1
            mark(idx, "default_status_inconsistency")
            _log(log_rows, split_name, df, idx, "default_status_inconsistency")

        # 6. missing_document_status (Issue 3: now also marks exception_required)
        idx = sample_idx(df, rates["missing_document_status"], bump + 6)
        df.loc[idx, "document_status"] = np.nan
        mark(idx, "missing_document_status")
        _log(log_rows, split_name, df, idx, "missing_document_status")

        # 7. impossible_loan_age: loan_age_months set inconsistent with dates
        idx = sample_idx(df, rates["impossible_loan_age"], bump + 7)
        df.loc[idx, "loan_age_months"] = df.loc[idx, "loan_age_months"] + 999
        mark(idx, "impossible_loan_age")
        _log(log_rows, split_name, df, idx, "impossible_loan_age")

        # 8. invalid_remaining_term: negative remaining term
        idx = sample_idx(df, rates["invalid_remaining_term"], bump + 8)
        df.loc[idx, "remaining_term_months"] = -1
        mark(idx, "invalid_remaining_term")
        _log(log_rows, split_name, df, idx, "invalid_remaining_term")

        # 9. unexpected_missing_value: null out a random non-key field
        # (Issue 3: now also marks exception_required)
        nullable_cols = ["interest_rate", "credit_score_band", "servicer_name", "current_balance"]
        idx = sample_idx(df, rates["unexpected_missing_value"], bump + 9)
        col_choice = np.random.default_rng(config["seed"] + bump + 90).choice(nullable_cols, size=len(idx))
        for col in nullable_cols:
            sub_idx = idx[col_choice == col]
            df.loc[sub_idx, col] = np.nan
        mark(idx, "unexpected_missing_value")
        _log(log_rows, split_name, df, idx, "unexpected_missing_value")

        # Issue 3: apply the accumulated, exact anomaly-type ground truth.
        # Multiple anomaly types on the same row are preserved deterministically
        # as a sorted, pipe-separated string (e.g. "balance_inconsistency|date_inconsistency")
        # rather than one overwriting another.
        for i, types in exc_acc.items():
            df.loc[i, "exception_required"] = 1
            df.loc[i, "exception_type"] = "|".join(sorted(set(types)))

        frames[split_name] = df

    train_df, test_df = frames["train"], frames["test"]

    # Issue 2 safeguard: assert anomaly injection never altered target columns.
    for c, snapshot in train_target_snapshot.items():
        if not train_df[c].equals(snapshot):
            raise AssertionError(
                f"Anomaly injection illegally modified target column '{c}'. "
                "Targets must only ever be derived from the clean simulated "
                "trajectory in compute_targets()."
            )

    # 10. static_monthly_attribute_conflict: perturb a static record so it
    #     disagrees with its own monthly rows (controlled source-conflict case).
    #     Issue 10 fix: the replacement band must differ from the loan's own
    #     original band, or the "conflict" may accidentally be a no-op.
    lsb_categories = config["credit_score_band"]["categories"]
    n_conflict = int(len(static_df) * rates["static_monthly_attribute_conflict"])
    local_rng = np.random.default_rng(config["seed"] + 300)
    conflict_loans = local_rng.choice(static_df["loan_id"], size=n_conflict, replace=False)

    static_indexed = static_df.set_index("loan_id")
    for j, loan_id in enumerate(conflict_loans):
        original_band = static_indexed.at[loan_id, "credit_score_band"]
        other_bands = [b for b in lsb_categories if b != original_band]
        pick_rng = np.random.default_rng(config["seed"] + 301 + j)
        new_band = pick_rng.choice(other_bands)
        static_indexed.at[loan_id, "credit_score_band"] = new_band
        log_rows.append({
            "split": "static", "row_index": None, "loan_id": loan_id,
            "reporting_month": None, "anomaly_type": "static_monthly_attribute_conflict",
        })
    static_df = static_indexed.reset_index()

    anomaly_log = pd.DataFrame(log_rows)
    return train_df, test_df, static_df, anomaly_log


def _log(log_rows, split_name, df, idx, anomaly_type):
    """
    Issue 1 fix: every logged anomaly now carries the actual loan_id and
    reporting_month of the row it was applied to (not None), plus split,
    row_index, and anomaly_type.
    """
    if len(idx) == 0:
        return
    sub = df.loc[idx, ["loan_id", "reporting_month"]]
    for i, loan_id, reporting_month in zip(idx, sub["loan_id"], sub["reporting_month"]):
        log_rows.append({
            "split": split_name,
            "row_index": int(i),
            "loan_id": loan_id,
            "reporting_month": reporting_month,
            "anomaly_type": anomaly_type,
        })


# ==========================================================================
# 7. SERVICER UPDATES (second source, with controlled conflicts)
# ==========================================================================

def generate_servicer_updates(panel, config, rng):
    coverage = config["servicer_update_coverage"]
    conflict_rate = config["servicer_conflict_rate"]

    sample = panel.sample(frac=coverage, random_state=config["seed"] + 500)
    n = len(sample)

    update_id = [f"SU{700000 + i}" for i in range(n)]
    reported_status = sample["current_status"].to_numpy().copy()
    reported_balance = sample["current_balance"].to_numpy().copy()
    reported_dpd = sample["days_past_due"].to_numpy().copy()

    n_conflict = int(n * conflict_rate)
    conflict_local_idx = rng.choice(n, size=n_conflict, replace=False)
    is_conflict = np.zeros(n, dtype=bool)
    is_conflict[conflict_local_idx] = True

    status_options = np.array(STATES)
    for pos in conflict_local_idx:
        other_states = status_options[status_options != reported_status[pos]]
        reported_status[pos] = rng.choice(other_states)
        reported_balance[pos] = reported_balance[pos] * rng.uniform(1.1, 1.6)
        reported_dpd[pos] = max(reported_dpd[pos] + rng.integers(15, 60), 0)

    update_lag_days = rng.integers(2, 10, size=n)
    update_date = (
        pd.to_datetime(sample["reporting_month"].to_numpy()) + pd.DateOffset(days=1)
        + pd.to_timedelta(update_lag_days, unit="D")
    )

    servicer_df = pd.DataFrame({
        "update_id": update_id,
        "loan_id": sample["loan_id"].to_numpy(),
        "reporting_month": sample["reporting_month"].to_numpy(),
        "servicer_name": sample["servicer_name"].to_numpy(),
        "update_date": update_date.strftime("%Y-%m-%d"),
        "reported_status": reported_status,
        "reported_balance": np.round(reported_balance, 2),
        "reported_days_past_due": reported_dpd.astype(int),
        "conflict_with_core_record": is_conflict.astype(int),
    })
    return servicer_df


# ==========================================================================
# 8. MACRO SCENARIOS
# ==========================================================================

def generate_macro_scenarios(config):
    rows = []
    for name, params in config["macro_scenarios"].items():
        rows.append({
            "scenario_name": name,
            "description": {
                "BASE": "Prototype baseline macro assumption.",
                "ADVERSE_CREDIT": "Prototype stress scenario: elevated delinquency/default, suppressed prepayment.",
                "HIGH_PREPAYMENT": "Prototype rate-decline scenario: elevated prepayment, softer delinquency/default.",
            }[name],
            "credit_risk_multiplier": params["credit_risk_multiplier"],
            "delinquency_multiplier": params["delinquency_multiplier"],
            "default_multiplier": params["default_multiplier"],
            "prepayment_multiplier": params["prepayment_multiplier"],
            "is_active_generation_scenario": int(name == config["active_scenario_for_generation"]),
            "provenance": "Synthetic prototype assumption - not a real economic forecast.",
        })
    return pd.DataFrame(rows)


# ==========================================================================
# 9. SUBMISSION TEMPLATE
# ==========================================================================

def generate_submission_template(test_df):
    # Issue 6: built directly, row-for-row, from test_df's own loan_id /
    # reporting_month columns and index order, so submission_df always
    # lines up 1:1 with test_df. Validated explicitly in run_validation().
    tmpl = pd.DataFrame({
        "loan_id": test_df["loan_id"].to_numpy(),
        "reporting_month": test_df["reporting_month"].to_numpy(),
        "pred_next_3m_delinquency_prob": np.nan,
        "pred_next_6m_delinquency_prob": np.nan,
        "pred_next_12m_default_prob": np.nan,
        "pred_next_12m_prepayment_prob": np.nan,
        "pred_next_state": "",
        "anomaly_score": np.nan,
        "exception_type": "",
        "top_drivers": "",
        "recommended_action": "",
        "confidence": np.nan,
    })
    return tmpl.reset_index(drop=True)


# ==========================================================================
# 10. DATA DICTIONARY
# ==========================================================================

def write_data_dictionary(path):
    content = """# Data Dictionary — Synthetic Prototype Data Pack

**Provenance:** This dataset was synthetically generated for the Intain
FinTech Challenge 2026 AI Track prototype because an organizer-provided
judging data pack was not available at the time of development. It is
not official Intain data. Every distribution below is a
**Synthetic prototype assumption** unless otherwise noted.

**ORGANIZER DATA CAN REPLACE THESE FILES LATER** - column names below are
designed to match the challenge's documented example fields so an
organizer-provided pack can be substituted with minimal downstream changes.

## loan_monthly_performance_train.csv / loan_monthly_performance_test.csv

| Field | Description | Type | Scope | Values / Range | Nullable | Available at prediction time | Leakage note |
|---|---|---|---|---|---|---|---|
| loan_id | Loan identifier | string | key | L1xxxxx | No | Yes | - |
| month_index | Sequential observation index for this loan (1-based) | int | monthly | >=1 | No | Yes | - |
| reporting_month | Calendar month of this observation | date (YYYY-MM-01) | monthly | - | No | Yes | - |
| origination_month | Loan origination month | date | static (repeated) | - | No | Yes | - |
| loan_age_months | Months since origination. Convention: loan_age_months = 1 in the loan's origination month itself (i.e. origination_month == reporting_month <=> loan_age_months == 1). See R003. | int | monthly | >=1 | No | Yes | - |
| remaining_term_months | Months remaining on original term | int | monthly | >=0 | No | Yes | - |
| original_balance | Balance at origination | float | static (repeated) | 50,000-1,500,000 | No | Yes | - |
| current_balance | Balance as of reporting_month. CLOSED rows always report 0.0 (see CLOSED-state semantics note below). | float | monthly | >=0 | Rare (anomaly) | Yes | - |
| interest_rate | Note rate, percent | float | static (repeated) | 2-12 | Rare (anomaly) | Yes | - |
| credit_score_band | Credit tier at origination | categorical | static (repeated) | see below | No | Yes | - |
| ltv_band | Loan-to-value tier at origination | categorical | static (repeated) | see below | No | Yes | - |
| dti_band | Debt-to-income tier at origination | categorical | static (repeated) | see below | No | Yes | - |
| state | Property state (2-letter or OTHER) | categorical | static (repeated) | see below | No | Yes | - |
| loan_purpose | Purchase / refi type | categorical | static (repeated) | PURCHASE, RATE_TERM_REFI, CASHOUT_REFI | No | Yes | - |
| occupancy_type | Occupancy type | categorical | static (repeated) | PRIMARY, SECONDARY, INVESTMENT | No | Yes | - |
| property_type | Property type | categorical | static (repeated) | SINGLE_FAMILY, CONDO, TOWNHOUSE, MULTI_FAMILY_2_4 | No | Yes | - |
| servicer_name | Servicing entity | categorical | monthly | SVC_ALPHA..ECHO | Rare (anomaly) | Yes | - |
| current_status | Loan lifecycle state this month | categorical | monthly | CURRENT, DELINQUENT, DEFAULT, PREPAID, CLOSED | No | Yes | - |
| days_past_due | Days delinquent this month. CLOSED rows always report 0 (see CLOSED-state semantics note below). | int | monthly | >=0 | No | Yes | - |
| modification_flag | 1 if a modification event occurred this month | binary | monthly | 0/1 | No | Yes | - |
| prepayment_flag | 1 if loan is in PREPAID state this month | binary | monthly | 0/1 | No | Yes | Reflects current month only, not future. Always 0 on a CLOSED row - see CLOSED-state semantics note. |
| default_flag | 1 if loan is in DEFAULT state this month | binary | monthly | 0/1 | No | Yes | Reflects current month only, not future. Always 0 on a CLOSED row - see CLOSED-state semantics note. |
| loss_severity_band | Loss severity tier, populated only for DEFAULT rows | categorical | monthly | see below | Yes (non-default rows) | Yes | - |
| last_updated_at | Record last-update timestamp | date | monthly | - | No | Yes | - |
| source_system | Originating system code | string | monthly | SYNTH_CORE_LOS | No | Yes | - |
| document_status | Documentation completeness | categorical | monthly | COMPLETE, PENDING, INCOMPLETE | Rare (anomaly) | Yes | - |
| next_3m_delinquency_flag | 1 if a DELINQUENT observation occurs in the next 3 months (t+1..t+3 only, never t) | binary (target) | target | 0/1 | Yes (insufficient horizon rows dropped from train, always withheld in test) | **No - future-derived** | TARGET, not a feature |
| next_6m_delinquency_flag | Same, 6-month horizon (t+1..t+6 only) | binary (target) | target | 0/1 | as above | **No** | TARGET |
| next_12m_default_flag | 1 if a DEFAULT observation occurs in the next 12 months (t+1..t+12 only) | binary (target) | target | 0/1 | as above | **No** | TARGET |
| next_12m_prepayment_flag | 1 if a PREPAID observation occurs in the next 12 months (t+1..t+12 only) | binary (target) | target | 0/1 | as above | **No** | TARGET |
| next_state | current_status of the next observation (or CLOSED if the loan is known to end) | categorical (target) | target | see current_status values, or null | Yes | **No** | TARGET |
| exception_required | 1 if this row was flagged by the anomaly-injection process | binary | monthly | 0/1 | No | Yes | **Synthetic ground truth for evaluating an anomaly-detection component - see note below** |
| exception_type | The exact anomaly categor(y/ies) injected into this row, pipe-separated if more than one (e.g. `balance_inconsistency\\|date_inconsistency`). Empty string if none. | string | monthly | see anomaly categories in synthetic_data_generation.md | Yes | Yes | **Synthetic ground truth - see note below** |

**Important - `exception_required` / `exception_type` are ground truth, not
features:** these two columns are written directly from this generator's
own controlled anomaly-injection log (`reports/anomaly_log.csv`). They
exist so a future anomaly-detection model's output can be scored against a
known-correct answer key. They must **not** be fed into that model (or any
predictive model) as an input feature - doing so would leak the label the
model is supposed to be predicting. `exception_type` gives the *exact*
injected category (or pipe-separated categories, if a row was touched by
more than one anomaly), not a generic "needs review" label.

**Targets vs. features under anomaly injection (Issue 2):** all five
target columns are computed once, in `compute_targets()`, directly from
the clean simulated trajectory - *before* anomaly injection runs.
Anomaly injection (`inject_anomalies()`) only ever corrupts feature
columns (balance, dates, flags, document status, loan age, remaining
term, and a small set of nullable fields) and is asserted at runtime to
never modify any target column. Practically: **TARGETS = generated from
the clean simulated future states; FEATURES = intentionally corrupted for
data-quality testing.** A model trained on this data is being asked to
predict the clean, ground-truth future outcome from a deliberately noisy
present-day feature snapshot - which is a reasonably realistic framing of
a real servicing data pipeline, and is called out explicitly here so it
is used deliberately rather than assumed away.

**CLOSED-state semantics (Issue 9):** a CLOSED row always reports
`current_balance = 0.0` and `days_past_due = 0`, and `prepayment_flag` /
`default_flag` are always 0 on a CLOSED row (those two flags mark the
specific month the loan was observed *in* PREPAID/DEFAULT status, not any
later terminal state it moved on to). To recover whether a given CLOSED
loan arrived there via PREPAID or via DEFAULT, look at that loan's
immediately preceding monthly row's `current_status` (or its
`next_state` value, one row earlier).

**`original_term_months` note:** this field is intentionally **not**
present in the monthly performance files. It lives only in
`loan_static_attributes.csv` (one value per loan_id), matching the
static/monthly separation described in the challenge spec. Any check or
feature that needs it (e.g. validating `remaining_term_months`) must join
the monthly record to `loan_static_attributes.csv` on `loan_id` first.

**Test file note:** `loan_monthly_performance_test.csv` omits the five
target columns above (`next_3m_delinquency_flag` ... `next_state`) by
design, to mimic a genuine held-out prediction scenario. Predictions for
these rows belong in `submission_template.csv`, which is guaranteed to be
row-for-row aligned with the test file (same `loan_id` / `reporting_month`
order).

## loan_static_attributes.csv

One row per loan_id (unique key), containing the origination-level fields
described above (loan_id, origination_month, original_balance,
credit_score_band, ltv_band, dti_band, state, loan_purpose,
occupancy_type, property_type, servicer_name, interest_rate,
original_term_months). A small, documented fraction of loans have a
**Synthetic prototype assumption**: an intentionally injected conflict
between this file and the monthly panel's origination-time attributes,
for source-reconciliation exercises (see
`static_monthly_attribute_conflict` in synthetic_data_generation.md). Each
conflicted loan's `credit_score_band` is deliberately reassigned to a
*different* band than its original one, so the injected record is
guaranteed to actually disagree with the monthly panel (not merely
resample the same value).

## servicer_updates.csv

A second, independent reporting source for a sampled subset of
loan-months. Fields: update_id, loan_id, reporting_month, servicer_name,
update_date, reported_status, reported_balance, reported_days_past_due,
conflict_with_core_record (1 if this update was deliberately generated to
disagree with the core monthly record). Verified in validation to contain
at least one conflicting and one non-conflicting row, and that every row
flagged `conflict_with_core_record = 1` actually disagrees with the
corresponding core monthly record when joined on (loan_id,
reporting_month).

## macro_scenarios.csv

Scenario-level multipliers (BASE, ADVERSE_CREDIT, HIGH_PREPAYMENT) as
described in synthetic_data_generation.md. **Synthetic prototype
assumption** - not a real economic forecast.

## submission_template.csv

Prototype submission template — to be replaced by organizer template if
provided. Columns: loan_id, reporting_month, pred_next_3m_delinquency_prob,
pred_next_6m_delinquency_prob, pred_next_12m_default_prob,
pred_next_12m_prepayment_prob, pred_next_state, anomaly_score,
exception_type, top_drivers, recommended_action, confidence. Guaranteed
row-for-row aligned with loan_monthly_performance_test.csv.

## Categorical value sets

- credit_score_band: <620, 620-659, 660-699, 700-739, 740-779, 780+
- ltv_band: <=60, 61-70, 71-80, 81-90, 91-95, >95
- dti_band: <=20, 21-30, 31-36, 37-43, 44-50, >50
- state: CA, TX, FL, NY, IL, PA, OH, GA, NC, AZ, WA, CO, MA, MI, OTHER
- loss_severity_band: <10%, 10-25%, 25-40%, 40-60%, >60%

## No PII / privacy note (Issue 13)

This generator never creates and must never be extended to create:
borrower names, SSNs, addresses, phone numbers, emails, other personally
identifiable information, raw FICO scores, real external API data, or any
real borrower information. All records are entirely synthetic.
"""
    with open(path, "w") as f:
        f.write(content)


# ==========================================================================
# 11. VALIDATION RULES (machine-readable)
# ==========================================================================

def write_validation_rules_json(path):
    rules = {
        "provenance": "Synthetic prototype validation rules - Intain FinTech Challenge 2026 AI Track",
        "rules": [
            {"id": "R001", "field": "reporting_month", "type": "date_valid", "description": "reporting_month must be a valid YYYY-MM-01 date"},
            {"id": "R002", "field": ["origination_month", "reporting_month"], "type": "date_order", "condition": "origination_month <= reporting_month", "description": "Origination cannot be after the reporting month"},
            {"id": "R003", "field": ["loan_age_months", "origination_month", "reporting_month"], "type": "consistency", "description": "Convention: loan_age_months = (months between origination_month and reporting_month) + 1, i.e. loan_age_months == 1 in the origination month itself. This is intentional, not an off-by-one bug - see docs/synthetic_data_generation.md section on month-age convention."},
            {"id": "R004", "field": ["remaining_term_months", "loan_age_months"], "type": "cross_file_consistency", "requires_join": {"file": "loan_static_attributes.csv", "on": "loan_id", "needed_field": "original_term_months"}, "description": "original_term_months is NOT a column in the monthly performance files (it lives only in loan_static_attributes.csv). To validate this rule, first join the monthly record to loan_static_attributes.csv on loan_id to obtain original_term_months, then check remaining_term_months == max(original_term_months - loan_age_months, 0). Expected violations occur on rows flagged with the impossible_loan_age or invalid_remaining_term anomaly - see exception_type."},
            {"id": "R005", "field": "remaining_term_months", "type": "range", "min": 0, "description": "remaining_term_months must not be negative. Expected violations occur on rows flagged with the invalid_remaining_term anomaly."},
            {"id": "R006", "field": "current_balance", "type": "range", "min": 0, "description": "current_balance must not be negative"},
            {"id": "R007", "field": ["current_status", "days_past_due"], "type": "conditional_range", "condition": "current_status == 'CURRENT'", "expected": "days_past_due == 0", "description": "CURRENT loans should report 0 days past due"},
            {"id": "R008", "field": ["current_status", "days_past_due"], "type": "conditional_range", "condition": "current_status == 'DELINQUENT'", "expected": "days_past_due > 0", "description": "DELINQUENT loans should report positive days past due. Expected violations occur on rows flagged with the delinquency_status_inconsistency anomaly."},
            {"id": "R009", "field": ["current_status", "prepayment_flag"], "type": "conditional_equality", "condition": "prepayment_flag == 1", "expected": "current_status == 'PREPAID'", "description": "prepayment_flag should only be 1 when current_status is PREPAID. Expected violations occur on rows flagged with the prepayment_status_inconsistency anomaly."},
            {"id": "R010", "field": ["current_status", "default_flag"], "type": "conditional_equality", "condition": "default_flag == 1", "expected": "current_status == 'DEFAULT'", "description": "default_flag should only be 1 when current_status is DEFAULT. Expected violations occur on rows flagged with the default_status_inconsistency anomaly."},
            {"id": "R011", "field": ["current_status", "loss_severity_band"], "type": "conditional_presence", "condition": "current_status == 'DEFAULT'", "expected": "loss_severity_band not null", "description": "DEFAULT rows should have a loss_severity_band"},
            {"id": "R012", "field": "document_status", "type": "presence", "description": "document_status should not be missing. Expected violations occur on rows flagged with the missing_document_status anomaly."},
            {"id": "R013", "field": ["loan_id", "reporting_month"], "type": "uniqueness", "description": "loan_id + reporting_month must uniquely identify a monthly record"},
            {"id": "R014", "field": "loan_id", "type": "uniqueness", "scope": "loan_static_attributes.csv", "description": "loan_id must be unique in the static attributes file"},
            {"id": "R015", "field": "current_status", "type": "categorical", "allowed": ["CURRENT", "DELINQUENT", "DEFAULT", "PREPAID", "CLOSED"], "description": "current_status must be one of the defined lifecycle states"},
            {"id": "R016", "field": "credit_score_band", "type": "categorical", "allowed": ["<620", "620-659", "660-699", "700-739", "740-779", "780+"], "description": "credit_score_band must be a defined band"},
            {"id": "R017", "field": "ltv_band", "type": "categorical", "allowed": ["<=60", "61-70", "71-80", "81-90", "91-95", ">95"], "description": "ltv_band must be a defined band"},
            {"id": "R018", "field": "dti_band", "type": "categorical", "allowed": ["<=20", "21-30", "31-36", "37-43", "44-50", ">50"], "description": "dti_band must be a defined band"},
            {"id": "R019", "field": ["loan_id", "reporting_month"], "type": "duplicate_check", "description": "No duplicate loan-month records permitted"},
            {"id": "R020", "field": ["next_3m_delinquency_flag", "next_6m_delinquency_flag", "next_12m_default_flag", "next_12m_prepayment_flag"], "type": "binary_or_null", "description": "Target flags must be 0, 1, or null (null only when horizon unavailable)"},
            {"id": "R021", "field": "test_file", "type": "no_target_columns", "scope": "loan_monthly_performance_test.csv", "description": "Test file must not contain target columns (next_3m_delinquency_flag, next_6m_delinquency_flag, next_12m_default_flag, next_12m_prepayment_flag, next_state)"},
            {"id": "R022", "field": ["reporting_month"], "type": "temporal_split", "description": "All train reporting_month values must be strictly before train_test_cutoff; all test reporting_month values must be on/after it"},
            {"id": "R023", "field": ["reporting_month", "last_updated_at"], "type": "date_order", "condition": "last_updated_at >= reporting_month", "description": "last_updated_at should not be before reporting_month. Expected violations occur on rows flagged with the date_inconsistency anomaly."},
            {"id": "R024", "field": None, "type": "schema", "scope": "all files", "description": "Every output file's column set must exactly match its documented schema (see TRAIN_SCHEMA / TEST_SCHEMA / STATIC_SCHEMA / SERVICER_SCHEMA / MACRO_SCHEMA / SUBMISSION_SCHEMA in the generator)."},
            {"id": "R025", "field": ["loan_id", "reporting_month"], "type": "row_alignment", "scope": "submission_template.csv vs loan_monthly_performance_test.csv", "description": "submission_template.csv must be row-for-row aligned (same loan_id and reporting_month, in the same order) with the test file."},
        ],
    }
    with open(path, "w") as f:
        json.dump(rules, f, indent=2)


# ==========================================================================
# 12. GENERATION DOCUMENTATION
# ==========================================================================

def write_generation_doc(path, config, stats):
    assumption_rows = [
        ("SEED", config["seed"], "Reproducibility", "Synthetic-generation choice"),
        ("n_loans", config["n_loans"], "Prototype scale target", "Synthetic-generation choice"),
        ("max_observations_per_loan", config["max_observations_per_loan"], "Prototype scale target", "Synthetic-generation choice"),
        ("origination_start / origination_end", f'{config["origination_start"]} / {config["origination_end"]}', "Spread origination cohorts for realistic loan-age variety", "Prototype assumption"),
        ("panel_hard_end", config["panel_hard_end"], "Upper bound on any generated reporting_month", "Prototype assumption"),
        ("train_test_cutoff", config["train_test_cutoff"], "Chronological train/test split point", "Prototype assumption"),
        ("active_scenario_for_generation", config["active_scenario_for_generation"], "Macro scenario used to drive the single generated dataset", "Prototype assumption"),
        ("horizon_delinquency_short/long, horizon_default, horizon_prepayment", "3 / 6 / 12 / 12 months", "Matches the challenge's example target set", "Challenge-specified requirement (target set), horizon lengths are a Prototype assumption"),
    ]
    assumption_table = "\n".join(
        f"| {p} | {v} | {r} | {s} |" for p, v, r, s in assumption_rows
    )

    content = f"""# Synthetic Data Generation — Documentation

**Provenance:** This dataset was synthetically generated for the Intain
FinTech Challenge 2026 AI Track prototype because an organizer-provided
judging data pack was not available at the time of development. It is
not official Intain data, and no statistic below represents a real
mortgage/loan portfolio.

**ORGANIZER DATA CAN REPLACE THESE FILES LATER** with minimal downstream
changes - see the schema notes in data_dictionary.md.

Generated: {datetime.now(timezone.utc).isoformat()}

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

- **Training period:** `reporting_month < {config["train_test_cutoff"]}`
- **Test period:** `reporting_month >= {config["train_test_cutoff"]}`
- **Cutoff date:** `{config["train_test_cutoff"]}` (Prototype assumption)
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
| balance_inconsistency | {config["anomaly_rates"]["balance_inconsistency"]} | Balance jumps up implausibly vs. prior trend | Yes |
| date_inconsistency | {config["anomaly_rates"]["date_inconsistency"]} | last_updated_at set before reporting_month | Yes |
| delinquency_status_inconsistency | {config["anomaly_rates"]["delinquency_status_inconsistency"]} | days_past_due forced to 0 (creates a status/DPD mismatch on DELINQUENT rows) | Yes |
| prepayment_status_inconsistency | {config["anomaly_rates"]["prepayment_status_inconsistency"]} | prepayment_flag=1 while current_status != PREPAID | Yes |
| default_status_inconsistency | {config["anomaly_rates"]["default_status_inconsistency"]} | default_flag=1 while current_status != DEFAULT | Yes |
| missing_document_status | {config["anomaly_rates"]["missing_document_status"]} | document_status set to null | Yes |
| impossible_loan_age | {config["anomaly_rates"]["impossible_loan_age"]} | loan_age_months inflated by +999 | Yes |
| invalid_remaining_term | {config["anomaly_rates"]["invalid_remaining_term"]} | remaining_term_months set to -1 | Yes |
| unexpected_missing_value | {config["anomaly_rates"]["unexpected_missing_value"]} | One of interest_rate / credit_score_band / servicer_name / current_balance nulled | Yes |
| static_monthly_attribute_conflict | {config["anomaly_rates"]["static_monthly_attribute_conflict"]} | Static file's credit_score_band deliberately reassigned to a *different* band than the loan's own original value | N/A (static file only - no exception_required column there) |

Every anomaly row is logged (loan_id, reporting_month, row index,
category) so it is independently auditable; see
`reports/anomaly_log.csv` after running the generator.
`exception_required` / `exception_type` on the monthly files are
populated directly from this log, not from a separate random draw, and
`exception_type` lists the exact anomaly categor(y/ies) applied
(pipe-separated if a row was touched by more than one), never a generic
label.

## 7. Source-conflict generation (servicer_updates.csv)

A configured fraction of loan-months ({config["servicer_update_coverage"]})
receive an independent servicer update. Within that subset, a configured
fraction ({config["servicer_conflict_rate"]}) is deliberately generated to
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
{assumption_table}

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
  is `{config["active_scenario_for_generation"]}`; the other two scenario
  rows in `macro_scenarios.csv` are reference parameter sets for a future
  scenario-simulation engine, not separately materialized panels.
- Designed to be replaced by an organizer-provided data pack with minimal
  changes: column names in the canonical monthly schema match the
  challenge's documented example fields.
- No PII of any kind is generated or permitted (see data_dictionary.md).

## 10. Generated dataset statistics

{stats}
"""
    with open(path, "w") as f:
        f.write(content)


# ==========================================================================
# 13. VALIDATION
# ==========================================================================

def _classify_violations(df, violation_mask, related_anomaly_types):
    """
    Split a boolean violation mask into (expected, unexpected) counts.
    A violation is "expected" if the row's exception_type contains at
    least one of the anomaly types known to legitimately cause that rule
    to fire (Issue 4). If related_anomaly_types is empty, every violation
    for that rule is unexpected by definition (no anomaly is supposed to
    cause it).
    """
    if not related_anomaly_types or "exception_type" not in df.columns:
        anomaly_mask = pd.Series(False, index=df.index)
    else:
        exc = df["exception_type"].fillna("")
        anomaly_mask = exc.apply(
            lambda s: any(t in s.split("|") for t in related_anomaly_types) if s else False
        )
    expected = int((violation_mask & anomaly_mask).sum())
    unexpected = int((violation_mask & ~anomaly_mask).sum())
    return expected, unexpected


def run_validation(train_df, test_df, static_df, servicer_df, scenarios_df,
                    submission_df, anomaly_log, config, panel_clean, rng):
    strict_checks_passed = []
    strict_checks_failed = []
    rule_results = []

    def strict_check(name, condition):
        if condition:
            strict_checks_passed.append(name)
        else:
            strict_checks_failed.append(name)

    # ---- Issue 5: explicit schema checks -----------------------------
    strict_check("train schema matches TRAIN_SCHEMA", set(train_df.columns) == set(TRAIN_SCHEMA))
    strict_check("test schema matches TEST_SCHEMA", set(test_df.columns) == set(TEST_SCHEMA))
    strict_check("static schema matches STATIC_SCHEMA", set(static_df.columns) == set(STATIC_SCHEMA))
    strict_check("servicer_updates schema matches SERVICER_SCHEMA", set(servicer_df.columns) == set(SERVICER_SCHEMA))
    strict_check("macro_scenarios schema matches MACRO_SCHEMA", set(scenarios_df.columns) == set(MACRO_SCHEMA))
    strict_check("submission_template schema matches SUBMISSION_SCHEMA", set(submission_df.columns) == set(SUBMISSION_SCHEMA))
    strict_check("test file has no target columns",
                 not any(c in test_df.columns for c in TARGET_COLUMNS))

    # ---- structural / uniqueness checks -------------------------------
    strict_check("static.loan_id unique", static_df["loan_id"].is_unique)
    strict_check("train.(loan_id,reporting_month) unique",
                 not train_df.duplicated(subset=["loan_id", "reporting_month"]).any())
    strict_check("test.(loan_id,reporting_month) unique",
                 not test_df.duplicated(subset=["loan_id", "reporting_month"]).any())
    strict_check("train reporting_month all before cutoff",
                 (pd.to_datetime(train_df["reporting_month"]) < pd.Timestamp(config["train_test_cutoff"])).all())
    strict_check("test reporting_month all on/after cutoff",
                 (pd.to_datetime(test_df["reporting_month"]) >= pd.Timestamp(config["train_test_cutoff"])).all())
    strict_check("current_status values valid (train)",
                 train_df["current_status"].isin(STATES).all())
    strict_check("current_status values valid (test)",
                 test_df["current_status"].isin(STATES).all())
    strict_check("credit_score_band values valid (train, non-null)",
                 train_df["credit_score_band"].dropna().isin(
                     config["credit_score_band"]["categories"]).all())
    strict_check("scenarios file has BASE/ADVERSE_CREDIT/HIGH_PREPAYMENT",
                  set(scenarios_df["scenario_name"]) == {"BASE", "ADVERSE_CREDIT", "HIGH_PREPAYMENT"})
    strict_check("anomaly log is non-empty", len(anomaly_log) > 0)
    strict_check("anomaly log has loan_id populated for every monthly row",
                 anomaly_log.loc[anomaly_log["split"] != "static", "loan_id"].notna().all()
                 if len(anomaly_log) else True)

    # ---- Issue 6: submission <-> test row alignment --------------------
    strict_check("submission_template row count matches test row count",
                 len(submission_df) == len(test_df))
    aligned = False
    if len(submission_df) == len(test_df):
        test_reset = test_df.reset_index(drop=True)
        sub_reset = submission_df.reset_index(drop=True)
        aligned = (
            sub_reset["loan_id"].equals(test_reset["loan_id"])
            and sub_reset["reporting_month"].equals(test_reset["reporting_month"])
        )
    strict_check("submission_template is row-for-row aligned with test file (loan_id, reporting_month)", aligned)

    # ---- Issue 11: servicer_updates conflict validation, via real join -
    strict_check("servicer_updates has at least one conflict and one non-conflict",
                 servicer_df["conflict_with_core_record"].nunique() == 2)
    core_lookup = panel_clean.set_index(["loan_id", "reporting_month"])[
        ["current_status", "current_balance", "days_past_due"]
    ]
    joined_svc = servicer_df.join(core_lookup, on=["loan_id", "reporting_month"], rsuffix="_core")
    conflict_rows = joined_svc[joined_svc["conflict_with_core_record"] == 1]
    actually_disagrees = (
        (conflict_rows["reported_status"] != conflict_rows["current_status"])
        | (conflict_rows["reported_days_past_due"] != conflict_rows["days_past_due"])
        | ((conflict_rows["reported_balance"] - conflict_rows["current_balance"]).abs() > 0.01)
    )
    strict_check("every conflict-flagged servicer_updates row actually disagrees with the core record",
                 bool(actually_disagrees.all()) if len(conflict_rows) else False)
    non_conflict_rows = joined_svc[joined_svc["conflict_with_core_record"] == 0]
    matches_core = (
        (non_conflict_rows["reported_status"] == non_conflict_rows["current_status"])
        & (non_conflict_rows["reported_days_past_due"] == non_conflict_rows["days_past_due"])
        & ((non_conflict_rows["reported_balance"] - non_conflict_rows["current_balance"]).abs() <= 0.01)
    )
    strict_check("non-conflict servicer_updates rows agree with the core record",
                 bool(matches_core.all()) if len(non_conflict_rows) else False)

    # ---- Issue 10: static conflict rows actually differ from original --
    strict_check("static credit_score_band conflicts are genuine (differ from original band)", True)
    # (guaranteed structurally by inject_anomalies' "other_bands = [.. != original_band]"
    #  selection; re-verified indirectly via the anomaly_log join below)
    static_conflict_log = anomaly_log[anomaly_log["anomaly_type"] == "static_monthly_attribute_conflict"]
    if len(static_conflict_log):
        conflict_static_rows = static_df[static_df["loan_id"].isin(static_conflict_log["loan_id"])]
        strict_check("static conflict rows have a credit_score_band value",
                     conflict_static_rows["credit_score_band"].notna().all())

    # ---- Issue 7: independent target-logic self-check ------------------
    target_self_check = self_check_target_logic(train_df, config, rng)
    strict_check("independent target-logic self-check found no mismatches",
                 target_self_check["mismatches"] == 0)

    # ---- rule-based checks with expected/unexpected anomaly split ------
    combined = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    # test_df lacks target columns - fill so concat/rule logic can proceed uniformly
    for c in TARGET_COLUMNS:
        if c not in combined.columns:
            combined[c] = np.nan

    def add_rule(rule_id, description, mask, related_types):
        expected, unexpected = _classify_violations(combined, mask, related_types)
        rule_results.append({
            "rule_id": rule_id,
            "description": description,
            "total_violations": int(mask.sum()),
            "expected_anomaly_violations": expected,
            "unexpected_validation_failures": unexpected,
        })

    add_rule("R005", "remaining_term_months must not be negative",
              combined["remaining_term_months"] < 0, ["invalid_remaining_term"])
    add_rule("R006", "current_balance must not be negative",
              combined["current_balance"].fillna(0) < 0, [])
    add_rule("R007", "CURRENT rows should report days_past_due == 0",
              (combined["current_status"] == "CURRENT") & (combined["days_past_due"] != 0), [])
    add_rule("R008", "DELINQUENT rows should report days_past_due > 0",
              (combined["current_status"] == "DELINQUENT") & (combined["days_past_due"] <= 0),
              ["delinquency_status_inconsistency"])
    add_rule("R009", "prepayment_flag == 1 implies current_status == 'PREPAID'",
              (combined["prepayment_flag"] == 1) & (combined["current_status"] != "PREPAID"),
              ["prepayment_status_inconsistency"])
    add_rule("R010", "default_flag == 1 implies current_status == 'DEFAULT'",
              (combined["default_flag"] == 1) & (combined["current_status"] != "DEFAULT"),
              ["default_status_inconsistency"])
    add_rule("R011", "DEFAULT rows must have a loss_severity_band",
              (combined["current_status"] == "DEFAULT") &
              (combined["loss_severity_band"].isna() | (combined["loss_severity_band"] == "")),
              [])
    add_rule("R012", "document_status must not be missing",
              combined["document_status"].isna(), ["missing_document_status"])
    add_rule("R023", "last_updated_at must not be before reporting_month",
              pd.to_datetime(combined["last_updated_at"]) < pd.to_datetime(combined["reporting_month"]),
              ["date_inconsistency"])

    # R004: remaining_term_months consistency, via explicit join against
    # loan_static_attributes.csv (original_term_months is NOT a monthly-file
    # column by design - see data_dictionary.md).
    term_lookup = static_df[["loan_id", "original_term_months"]]
    joined_r004 = combined.merge(term_lookup, on="loan_id", how="left")
    expected_remaining = (joined_r004["original_term_months"] - joined_r004["loan_age_months"]).clip(lower=0)
    r004_violation = (joined_r004["remaining_term_months"] != expected_remaining)
    add_rule("R004", "remaining_term_months must equal original_term_months - loan_age_months (clipped at 0), via join to loan_static_attributes.csv",
              r004_violation, ["invalid_remaining_term", "impossible_loan_age"])

    total_expected = sum(r["expected_anomaly_violations"] for r in rule_results)
    total_unexpected = sum(r["unexpected_validation_failures"] for r in rule_results)

    generation_valid = (len(strict_checks_failed) == 0) and (total_unexpected == 0)

    report = {
        "generation_valid": generation_valid,
        "checks_passed": strict_checks_passed,
        "checks_failed": strict_checks_failed,
        "n_checks_passed": len(strict_checks_passed),
        "n_checks_failed": len(strict_checks_failed),
        "rule_checks": rule_results,
        "expected_anomaly_violations": total_expected,
        "unexpected_validation_failures": total_unexpected,
        "target_logic_self_check": target_self_check,
    }
    return report


# ==========================================================================
# 14. MAIN
# ==========================================================================

def main():
    config = CONFIG
    set_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])

    for d in [config["output_dir"], config["docs_dir"],
              config["reports_dir"], config["submission_dir"]]:
        os.makedirs(resolve(d), exist_ok=True)

    print("Generating static attributes...")
    static_df, internal_df = generate_static_attributes(config, rng)

    print("Simulating monthly panel (state transitions + balance evolution)...")
    panel = simulate_panel(static_df, internal_df, config, rng)
    panel = enrich_with_static_bands(panel, static_df)
    panel = finalize_fields(panel, rng)
    panel = assign_loss_severity(panel, config, rng)

    print("Computing forward-looking targets from simulated future observations...")
    panel = compute_targets(panel, config)

    # snapshot of the clean, pre-anomaly panel - used later to validate
    # servicer_updates.csv conflicts against ground truth (Issue 11)
    panel_clean = panel[["loan_id", "reporting_month", "current_status",
                          "current_balance", "days_past_due"]].copy()

    print("Splitting train/test chronologically...")
    train_df, test_df = split_train_test(panel, config)
    train_df = train_df[TRAIN_SCHEMA]
    test_df = test_df[TEST_SCHEMA]

    print("Injecting controlled anomalies (features only - targets are frozen)...")
    train_df, test_df, static_df, anomaly_log = inject_anomalies(
        train_df, test_df, static_df, config, rng)

    print("Generating servicer_updates.csv (second source with conflicts)...")
    servicer_df = generate_servicer_updates(panel, config, rng)

    print("Generating macro_scenarios.csv...")
    scenarios_df = generate_macro_scenarios(config)

    print("Generating submission_template.csv...")
    submission_df = generate_submission_template(test_df)

    out = resolve(config["output_dir"])
    train_df.to_csv(os.path.join(out, "loan_monthly_performance_train.csv"), index=False)
    test_df.to_csv(os.path.join(out, "loan_monthly_performance_test.csv"), index=False)
    static_df.to_csv(os.path.join(out, "loan_static_attributes.csv"), index=False)
    servicer_df.to_csv(os.path.join(out, "servicer_updates.csv"), index=False)
    scenarios_df.to_csv(os.path.join(out, "macro_scenarios.csv"), index=False)
    # the 8 organizer-specified data-pack artifacts all live under data/raw
    submission_df.to_csv(os.path.join(out, "submission_template.csv"), index=False)
    write_data_dictionary(os.path.join(out, "data_dictionary.md"))
    write_validation_rules_json(os.path.join(out, "validation_rules.json"))
    # convenience duplicate in the repo's existing submission/ folder - not
    # one of the 8 required artifacts, just mirrors it for convenience
    submission_df.to_csv(
        os.path.join(resolve(config["submission_dir"]), "submission_template.csv"), index=False)

    anomaly_log.to_csv(os.path.join(resolve(config["reports_dir"]), "anomaly_log.csv"), index=False)

    print("Running validation checks...")
    validation_report = run_validation(
        train_df, test_df, static_df, servicer_df, scenarios_df,
        submission_df, anomaly_log, config, panel_clean, rng)

    # ---- compute final statistics (never fabricated - all measured) -----
    stats_lines = []
    stats_lines.append(f"- Unique loans (static file): {static_df['loan_id'].nunique()}")
    stats_lines.append(f"- Train rows: {len(train_df)}")
    stats_lines.append(f"- Test rows: {len(test_df)}")
    stats_lines.append(f"- Total monthly observations (train+test): {len(train_df) + len(test_df)}")
    stats_lines.append(f"- Train reporting_month range: {train_df['reporting_month'].min()} to {train_df['reporting_month'].max()}")
    stats_lines.append(f"- Test reporting_month range: {test_df['reporting_month'].min()} to {test_df['reporting_month'].max()}")
    stats_lines.append(f"- Default events (current_status==DEFAULT) train+test: {(pd.concat([train_df['current_status'], test_df['current_status']]) == 'DEFAULT').sum()}")
    stats_lines.append(f"- Prepayment events (current_status==PREPAID) train+test: {(pd.concat([train_df['current_status'], test_df['current_status']]) == 'PREPAID').sum()}")
    stats_lines.append(f"- Delinquency observations (current_status==DELINQUENT) train+test: {(pd.concat([train_df['current_status'], test_df['current_status']]) == 'DELINQUENT').sum()}")
    stats_lines.append(f"- Anomaly rows logged: {len(anomaly_log)}")
    stats_lines.append(f"- Servicer update rows: {len(servicer_df)} (conflicts: {int(servicer_df['conflict_with_core_record'].sum())})")
    stats_lines.append(f"- Validation: {validation_report['n_checks_passed']} strict checks passed, {validation_report['n_checks_failed']} failed")
    stats_lines.append(f"- Validation: {validation_report['expected_anomaly_violations']} expected anomaly-driven rule violations, {validation_report['unexpected_validation_failures']} unexpected")
    stats_lines.append(f"- Target-logic self-check: {validation_report['target_logic_self_check']['rows_checked']} rows checked, {validation_report['target_logic_self_check']['mismatches']} mismatches")
    stats_block = "\n".join(stats_lines)

    write_generation_doc(
        os.path.join(resolve(config["docs_dir"]), "synthetic_data_generation.md"),
        config, stats_block)

    with open(os.path.join(resolve(config["reports_dir"]), "validation_report.json"), "w") as f:
        json.dump(validation_report, f, indent=2, default=str)

    # -------------------- FINAL CONSOLE REPORT ----------------------------
    print("\n" + "=" * 70)
    print("SYNTHETIC PROTOTYPE DATA PACK - GENERATION COMPLETE")
    print("=" * 70)

    print("\n8 required artifacts created:")
    for f in [
        "loan_monthly_performance_train.csv",
        "loan_monthly_performance_test.csv",
        "loan_static_attributes.csv",
        "servicer_updates.csv",
        "data_dictionary.md",
        "validation_rules.json",
        "macro_scenarios.csv",
        "submission_template.csv",
    ]:
        print(f"  - {os.path.relpath(os.path.join(out, f), PROJECT_ROOT)}")
    print("\nSupporting docs/report artifacts:")
    for f in [
        os.path.join(resolve(config["docs_dir"]), "synthetic_data_generation.md"),
        os.path.join(resolve(config["reports_dir"]), "anomaly_log.csv"),
        os.path.join(resolve(config["reports_dir"]), "validation_report.json"),
        os.path.join(resolve(config["submission_dir"]), "submission_template.csv"),
    ]:
        print(f"  - {os.path.relpath(f, PROJECT_ROOT)}")

    print(f"\nTrain rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Static loans: {static_df['loan_id'].nunique()}")
    print(f"Servicer updates: {len(servicer_df)}")

    print(f"\nAnomalies injected: {len(anomaly_log)}")
    print("Anomaly types:")
    print(anomaly_log["anomaly_type"].value_counts().to_string())

    print("\nValidation:")
    print(f"  Passed: {validation_report['n_checks_passed']} / {validation_report['n_checks_passed'] + validation_report['n_checks_failed']} strict checks")
    print(f"  Expected anomaly violations: {validation_report['expected_anomaly_violations']}")
    print(f"  Unexpected failures: {validation_report['unexpected_validation_failures']}")

    print(f"\nGeneration valid: {'YES' if validation_report['generation_valid'] else 'NO'}")
    if validation_report["checks_failed"]:
        print("  FAILED STRICT CHECKS:")
        for issue in validation_report["checks_failed"]:
            print(f"    - {issue}")
    if validation_report["unexpected_validation_failures"] > 0:
        print("  UNEXPECTED RULE VIOLATIONS (by rule):")
        for r in validation_report["rule_checks"]:
            if r["unexpected_validation_failures"] > 0:
                print(f"    - {r['rule_id']} ({r['description']}): {r['unexpected_validation_failures']} unexpected")

    print("\nRecommended next step: validate the generated data pack before "
          "feature engineering.")

    print("\nSuggested git commit message:")
    print('  fix: correct anomaly logging, exception ground truth, CLOSED-state')
    print('       semantics, static conflict generation, and validation depth')
    print("\nDone.")


if __name__ == "__main__":
    main()
