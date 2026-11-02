"""Data loading, validation, cleaning and train/test splitting."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from credit_scoring.schema import (
    CLASS_LABELS,
    GROUP_COL,
    LABEL_TO_ID,
    LOAN_TYPE_COLS,
    REQUIRED_RAW_COLS,
    TARGET,
)

logger = logging.getLogger(__name__)

# The source dataset uses 219 months as a sentinel when the free-text
# ``credit_history_age`` field was missing.
CREDIT_HISTORY_SENTINEL = 219
MAX_PLAUSIBLE_AGE = 90


class DataValidationError(ValueError):
    """Raised when an input frame does not match the expected schema."""


def load_raw(path: Path) -> pd.DataFrame:
    """Load the raw CSV, lower-case headers and validate the schema.

    Args:
        path: Location of the CSV file.

    Returns:
        Raw records with lower-cased column names.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        DataValidationError: If the file is empty, misses required columns
            or contains unknown target labels.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Set CREDIT_DATA_PATH in .env.")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"Dataset is empty: {path}") from exc
    df.columns = df.columns.str.strip().str.lower()
    validate_raw(df)
    logger.info("Loaded %d rows x %d cols from %s", *df.shape, path)
    return df


def validate_raw(df: pd.DataFrame) -> None:
    """Check that ``df`` has the required columns and valid target labels.

    Raises:
        DataValidationError: On any schema violation.
    """
    if df.empty:
        raise DataValidationError("Dataset has no rows.")
    missing = sorted(set(REQUIRED_RAW_COLS) - set(df.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    unknown = set(df[TARGET].dropna().unique()) - set(CLASS_LABELS)
    if unknown:
        raise DataValidationError(f"Unknown {TARGET} labels: {sorted(unknown)}")


def _impute_credit_history(df: pd.DataFrame) -> pd.Series:
    """Rebuild sentinel ``credit_history_age_months`` values per customer.

    Credit history grows by one month per record month, so
    ``months - record_month`` is constant for a customer. We estimate that
    offset from the customer's valid rows and add it back to ``record_month``.
    Customers with no valid row get ``NaN`` (and are dropped later).

    Note: the original notebook assigned ``record_month`` itself here, which
    collapsed ~8k credit histories to values between 1 and 8.
    """
    months = df["credit_history_age_months"].astype(float)
    is_sentinel = df["credit_history_age"].isna() & (months == CREDIT_HISTORY_SENTINEL)
    valid_offset = (months - df["record_month"]).where(~is_sentinel)
    offset = valid_offset.groupby(df[GROUP_COL]).transform("median")
    return months.where(~is_sentinel, df["record_month"] + offset)


def _fix_age_outliers(df: pd.DataFrame) -> pd.Series:
    """Replace implausible ages with the customer's typical (median) valid age."""
    age = df["customer_age"].astype(float)
    valid = age.where(age <= MAX_PLAUSIBLE_AGE)
    customer_age = valid.groupby(df[GROUP_COL]).transform("median").round()
    return valid.fillna(customer_age)


def _drop_income_typos(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose income is above the IQR fence *and* appears <= 3 times.

    Real high earners repeat their income across monthly records; one-off
    extreme values are data-entry errors.
    """
    income = df["annual_income"]
    q1, q3 = income.quantile([0.25, 0.75])
    fence = q3 + 1.5 * (q3 - q1)
    counts = income.map(income.value_counts())
    return df[~((income >= fence) & (counts <= 3))]


def _mode_or_nan(values: pd.Series) -> object:
    mode = values.mode()
    return mode.iloc[0] if not mode.empty else np.nan


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning pipeline to a raw frame.

    Steps: normalise loan-type labels, repair credit-history sentinels, fix
    age outliers, drop income typos, drop rows still missing key fields,
    impute occupation per customer and binarise ``min_amount_payment``.
    ``customer_id`` is kept for group-aware splitting.

    Args:
        df: Output of :func:`load_raw`.

    Returns:
        A new, cleaned frame with a fresh ``RangeIndex``.
    """
    out = df.copy()
    out[LOAN_TYPE_COLS] = out[LOAN_TYPE_COLS].replace(r"^\s*and\s+", "", regex=True)
    out["credit_history_age_months"] = _impute_credit_history(out)
    out["customer_age"] = _fix_age_outliers(out)
    out = _drop_income_typos(out)
    out = out.dropna(subset=["credit_history_age_months", "payment_behavior", "customer_age"])

    occupation_mode = out.groupby(GROUP_COL)["customer_occupation"].transform(_mode_or_nan)
    out["customer_occupation"] = (
        out["customer_occupation"].fillna(occupation_mode).fillna("Unknown")
    )
    out["min_amount_payment"] = (out["min_amount_payment"] == "Yes").astype(int)

    out = out.drop(columns=["credit_history_age", "record_id"])
    logger.info("Cleaning kept %d / %d rows", len(out), len(df))
    return out.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the encoded target.

    Returns:
        ``(X, y)`` where ``y`` holds integer class ids (Poor=0, Standard=1, Good=2).
    """
    y = df[TARGET].map(LABEL_TO_ID).astype(int)
    X = df.drop(columns=[TARGET, GROUP_COL], errors="ignore")
    return X, y


def train_test_split_df(
    df: pd.DataFrame,
    *,
    strategy: str = "group",
    test_size: float = 0.2,
    random_state: int = 108,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a cleaned frame into train and test sets.

    Args:
        df: Cleaned frame containing ``customer_id`` and ``credit_score``.
        strategy: ``"group"`` keeps all records of a customer on one side
            (no leakage, recommended); ``"row"`` reproduces the original
            notebook's stratified row-level split.
        test_size: Fraction of data held out for testing.
        random_state: Seed for reproducibility.

    Returns:
        ``(train_df, test_df)``.

    Raises:
        ValueError: On an unknown strategy or a ``test_size`` outside (0, 1).
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if strategy == "row":
        train, test = train_test_split(
            df, test_size=test_size, stratify=df[TARGET], random_state=random_state
        )
        return train, test
    if strategy == "group":
        n_splits = max(2, round(1 / test_size))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, df[TARGET], groups=df[GROUP_COL]))
        return df.iloc[train_idx], df.iloc[test_idx]
    raise ValueError(f"Unknown split strategy: {strategy!r}")
