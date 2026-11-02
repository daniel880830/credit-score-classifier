"""Column names and category vocabularies shared across the pipeline."""

from __future__ import annotations

TARGET = "credit_score"
GROUP_COL = "customer_id"

# Ordinal encoding of the target: higher = better credit.
CLASS_LABELS: tuple[str, ...] = ("Poor", "Standard", "Good")
LABEL_TO_ID: dict[str, int] = {label: i for i, label in enumerate(CLASS_LABELS)}

LOAN_TYPE_COLS: list[str] = ["loan_types"] + [f"loan_types_{i}" for i in range(2, 10)]

NUMERIC_COLS: list[str] = [
    "record_month",
    "customer_age",
    "annual_income",
    "monthly_salary",
    "bank_accounts",
    "credit_cards",
    "interest_rate",
    "loans",
    "payment_delay_days",
    "delayed_payments",
    "credit_limit_changes",
    "credit_inquiries",
    "outstanding_debt",
    "credit_utilization",
    "credit_history_age_months",
    "min_amount_payment",
    "monthly_emi",
    "monthly_investment",
    "monthly_balance",
]

CREDIT_MIX_ORDER: list[str] = ["Bad", "Standard", "Good"]
NOMINAL_COLS: list[str] = ["customer_occupation", "payment_behavior"]

FEATURE_COLS: list[str] = NUMERIC_COLS + ["credit_mix"] + NOMINAL_COLS + LOAN_TYPE_COLS

# Columns that must exist in the raw CSV (after lower-casing headers).
REQUIRED_RAW_COLS: list[str] = [
    "record_id",
    GROUP_COL,
    "credit_history_age",
    *FEATURE_COLS,
    TARGET,
]

LOAN_TYPES: tuple[str, ...] = (
    "Auto Loan",
    "Credit-Builder Loan",
    "Debt Consolidation Loan",
    "Home Equity Loan",
    "Mortgage Loan",
    "Not Specified",
    "Payday Loan",
    "Personal Loan",
    "Student Loan",
)

PAYMENT_BEHAVIORS: tuple[str, ...] = (
    "High_spent_Large_value_payments",
    "High_spent_Medium_value_payments",
    "High_spent_Small_value_payments",
    "Low_spent_Large_value_payments",
    "Low_spent_Medium_value_payments",
    "Low_spent_Small_value_payments",
)
