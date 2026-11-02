"""Shared fixtures: a small synthetic dataset that mirrors the real schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_scoring.config import Settings
from credit_scoring.schema import LOAN_TYPES, PAYMENT_BEHAVIORS

OCCUPATIONS = ["Scientist", "Teacher", "Engineer", "Lawyer"]

FAST_PARAMS_TOML = """
[knn.params]
n_neighbors = 3
weights = "distance"
metric = "l1"

[knn.grid]
clf__n_neighbors = [3, 5]

[xgboost.params]
n_estimators = 20
max_depth = 3
learning_rate = 0.3
tree_method = "hist"
n_jobs = 1

[random_forest.params]
n_estimators = 20
max_depth = 0
n_jobs = 1
"""


def make_raw_frame(n_customers: int = 60, months: int = 8, seed: int = 0) -> pd.DataFrame:
    """Build raw records in the CSV's original (Title_Case) header format.

    The label is a deterministic function of debt and payment delay so models
    can learn it, and every customer has all three classes spread evenly.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        income = float(rng.integers(20_000, 120_000))
        history_start = int(rng.integers(24, 300))
        n_loans = int(rng.integers(0, 4))
        loans = list(rng.choice(LOAN_TYPES, size=n_loans, replace=False))
        loans = [loans[0], *[f"and {x}" for x in loans[1:]]] if loans else []
        for m in range(1, months + 1):
            debt = float(rng.uniform(0, 5000))
            delay = int(rng.integers(0, 60))
            risk = debt / 5000 + delay / 60
            label = "Good" if risk < 0.7 else "Standard" if risk < 1.3 else "Poor"
            row = {
                "Record_ID": f"0x{c:04x}{m}",
                "Customer_ID": f"CUS_{c:04d}",
                "Record_Month": m,
                "Customer_Age": int(rng.integers(20, 60)),
                "Customer_Occupation": OCCUPATIONS[c % len(OCCUPATIONS)],
                "Annual_Income": income,
                "Monthly_Salary": income / 12,
                "Bank_Accounts": float(rng.integers(1, 8)),
                "Credit_Cards": int(rng.integers(1, 8)),
                "Interest_Rate": int(rng.integers(1, 30)),
                "Loans": n_loans,
                "Payment_Delay_Days": delay,
                "Delayed_Payments": float(rng.integers(0, 20)),
                "Credit_Limit_Changes": float(rng.uniform(-5, 20)),
                "Credit_Inquiries": float(rng.integers(0, 10)),
                "Credit_Mix": ["Bad", "Standard", "Good"][c % 3],
                "Outstanding_Debt": debt,
                "Credit_Utilization": float(rng.uniform(20, 45)),
                "Credit_History_Age": f"{(history_start + m) // 12} Years",
                "Credit_History_Age_Months": float(history_start + m),
                "Min_Amount_Payment": "Yes" if c % 2 else "No",
                "Monthly_EMI": float(rng.uniform(0, 300)),
                "Monthly_Investment": float(rng.uniform(0, 500)),
                "Payment_Behavior": PAYMENT_BEHAVIORS[c % len(PAYMENT_BEHAVIORS)],
                "Monthly_Balance": float(rng.uniform(100, 900)),
                "Credit_Score": label,
            }
            for i in range(9):
                col = "Loan_Types" if i == 0 else f"Loan_Types_{i + 1}"
                row[col] = loans[i] if i < len(loans) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def lower(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case headers the way ``load_raw`` does."""
    out = df.copy()
    out.columns = out.columns.str.lower()
    return out


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Raw synthetic frame with lower-cased headers."""
    return lower(make_raw_frame())


@pytest.fixture(scope="session")
def project_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temp project layout with a synthetic CSV and fast hyperparameters."""
    root = tmp_path_factory.mktemp("project")
    (root / "data").mkdir()
    make_raw_frame(n_customers=90).to_csv(root / "data" / "records.csv", index=False)
    (root / "params.toml").write_text(FAST_PARAMS_TOML, encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def settings(project_dir: Path) -> Settings:
    """Settings pointing at the temp project (ignores any local .env)."""
    return Settings(
        _env_file=None,
        data_path=project_dir / "data" / "records.csv",
        model_dir=project_dir / "models",
        reports_dir=project_dir / "reports",
        params_path=project_dir / "params.toml",
        model_name="random_forest",
        api_max_batch_size=5,
    )


@pytest.fixture
def valid_payload() -> dict:
    """A valid /predict request body."""
    return {
        "record_month": 3,
        "customer_age": 35,
        "customer_occupation": "Engineer",
        "annual_income": 65000.0,
        "monthly_salary": 5400.0,
        "bank_accounts": 3,
        "credit_cards": 4,
        "interest_rate": 12,
        "loans": 2,
        "loan_types": ["Auto Loan", "and Student Loan"],
        "payment_delay_days": 10,
        "delayed_payments": 4,
        "credit_limit_changes": 6.5,
        "credit_inquiries": 3,
        "credit_mix": "Standard",
        "outstanding_debt": 1200.0,
        "credit_utilization": 31.2,
        "credit_history_age_months": 180,
        "min_amount_payment": False,
        "monthly_emi": 90.0,
        "monthly_investment": 150.0,
        "payment_behavior": "Low_spent_Small_value_payments",
        "monthly_balance": 400.0,
    }
