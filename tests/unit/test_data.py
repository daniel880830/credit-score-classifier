from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_scoring.data import (
    CREDIT_HISTORY_SENTINEL,
    DataValidationError,
    clean,
    load_raw,
    split_features_target,
    train_test_split_df,
    validate_raw,
)
from credit_scoring.schema import FEATURE_COLS, GROUP_COL, TARGET
from tests.conftest import make_raw_frame

# --------------------------------------------------------------------- loading


def test_load_raw_lowercases_headers(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    make_raw_frame(n_customers=3).to_csv(path, index=False)
    df = load_raw(path)
    assert all(c == c.lower() for c in df.columns)
    assert len(df) == 3 * 8


def test_load_raw_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="CREDIT_DATA_PATH"):
        load_raw(tmp_path / "nope.csv")


def test_load_raw_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DataValidationError, match="empty"):
        load_raw(path)


def test_load_raw_header_only(tmp_path: Path) -> None:
    path = tmp_path / "header.csv"
    make_raw_frame(n_customers=1).head(0).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="no rows"):
        load_raw(path)


def test_validate_raw_missing_columns(raw_df: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="annual_income"):
        validate_raw(raw_df.drop(columns=["annual_income"]))


def test_validate_raw_unknown_label(raw_df: pd.DataFrame) -> None:
    raw_df.loc[0, TARGET] = "Excellent"
    with pytest.raises(DataValidationError, match="Excellent"):
        validate_raw(raw_df)


# -------------------------------------------------------------------- cleaning


def test_clean_does_not_mutate_input(raw_df: pd.DataFrame) -> None:
    before = raw_df.copy()
    clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_clean_strips_and_prefix_from_loan_types(raw_df: pd.DataFrame) -> None:
    out = clean(raw_df)
    values = out.filter(like="loan_types").stack()
    assert not values.str.startswith("and ").any()


def test_clean_imputes_credit_history_sentinel_from_same_customer(raw_df: pd.DataFrame) -> None:
    customer = raw_df[GROUP_COL].iloc[0]
    rows = raw_df.index[raw_df[GROUP_COL] == customer]
    expected = raw_df.loc[rows[3], "credit_history_age_months"]
    raw_df.loc[rows[3], "credit_history_age"] = np.nan
    raw_df.loc[rows[3], "credit_history_age_months"] = CREDIT_HISTORY_SENTINEL

    out = clean(raw_df)
    got = out.loc[
        (out[GROUP_COL] == customer) & (out["record_month"] == raw_df.loc[rows[3], "record_month"]),
        "credit_history_age_months",
    ]
    assert got.item() == expected


def test_clean_drops_customer_with_only_sentinel_history(raw_df: pd.DataFrame) -> None:
    customer = raw_df[GROUP_COL].iloc[0]
    mask = raw_df[GROUP_COL] == customer
    raw_df.loc[mask, "credit_history_age"] = np.nan
    raw_df.loc[mask, "credit_history_age_months"] = CREDIT_HISTORY_SENTINEL
    out = clean(raw_df)
    assert customer not in set(out[GROUP_COL])


def test_clean_keeps_genuine_219_month_history(raw_df: pd.DataFrame) -> None:
    # 219 is only a sentinel when the free-text field is missing.
    raw_df.loc[0, "credit_history_age_months"] = CREDIT_HISTORY_SENTINEL
    out = clean(raw_df)
    assert (out["credit_history_age_months"] == CREDIT_HISTORY_SENTINEL).any()


def test_clean_replaces_age_outlier_with_customer_median(raw_df: pd.DataFrame) -> None:
    customer = raw_df[GROUP_COL].iloc[0]
    mask = raw_df[GROUP_COL] == customer
    raw_df.loc[mask, "customer_age"] = 30
    raw_df.loc[raw_df.index[mask][2], "customer_age"] = 500
    out = clean(raw_df)
    assert (out.loc[out[GROUP_COL] == customer, "customer_age"] == 30).all()


def test_clean_drops_age_outlier_without_valid_neighbour(raw_df: pd.DataFrame) -> None:
    customer = raw_df[GROUP_COL].iloc[0]
    raw_df.loc[raw_df[GROUP_COL] == customer, "customer_age"] = 150
    out = clean(raw_df)
    assert customer not in set(out[GROUP_COL])
    assert out["customer_age"].max() <= 90


def test_clean_drops_one_off_income_typos_but_keeps_repeated_high_income(
    raw_df: pd.DataFrame,
) -> None:
    rich = raw_df[GROUP_COL].iloc[0]
    raw_df.loc[raw_df[GROUP_COL] == rich, "annual_income"] = 5_000_000.0  # 8 repeats -> real
    raw_df.loc[raw_df.index[-1], "annual_income"] = 9_999_999.0  # single value -> typo
    out = clean(raw_df)
    assert (out["annual_income"] == 5_000_000.0).sum() == 8
    assert (out["annual_income"] == 9_999_999.0).sum() == 0


def test_clean_drops_missing_payment_behavior(raw_df: pd.DataFrame) -> None:
    raw_df.loc[0:4, "payment_behavior"] = np.nan
    out = clean(raw_df)
    assert out["payment_behavior"].notna().all()
    assert len(out) == len(raw_df) - 5


def test_clean_fills_occupation_by_customer_mode_then_unknown(raw_df: pd.DataFrame) -> None:
    first, second = raw_df[GROUP_COL].unique()[:2]
    raw_df.loc[raw_df.index[raw_df[GROUP_COL] == first][0], "customer_occupation"] = np.nan
    raw_df.loc[raw_df[GROUP_COL] == second, "customer_occupation"] = np.nan
    out = clean(raw_df)
    assert out["customer_occupation"].notna().all()
    assert set(out.loc[out[GROUP_COL] == second, "customer_occupation"]) == {"Unknown"}
    assert out.loc[out[GROUP_COL] == first, "customer_occupation"].nunique() == 1


def test_clean_binarises_min_amount_payment_and_drops_ids(raw_df: pd.DataFrame) -> None:
    out = clean(raw_df)
    assert set(out["min_amount_payment"].unique()) <= {0, 1}
    assert {"record_id", "credit_history_age"}.isdisjoint(out.columns)
    assert GROUP_COL in out.columns  # kept for grouped splitting
    assert out.index.equals(pd.RangeIndex(len(out)))


# ------------------------------------------------------------------- splitting


def test_split_features_target_encodes_labels(raw_df: pd.DataFrame) -> None:
    X, y = split_features_target(clean(raw_df))
    assert set(y.unique()) <= {0, 1, 2}
    assert GROUP_COL not in X.columns and TARGET not in X.columns
    assert set(FEATURE_COLS) <= set(X.columns)


def test_group_split_has_no_customer_leakage(raw_df: pd.DataFrame) -> None:
    train, test = train_test_split_df(clean(raw_df), strategy="group", test_size=0.2)
    assert set(train[GROUP_COL]).isdisjoint(test[GROUP_COL])
    assert 0.1 < len(test) / (len(train) + len(test)) < 0.3


def test_row_split_is_stratified_and_reproducible(raw_df: pd.DataFrame) -> None:
    df = clean(raw_df)
    train_a, test_a = train_test_split_df(df, strategy="row", random_state=1)
    _, test_b = train_test_split_df(df, strategy="row", random_state=1)
    pd.testing.assert_frame_equal(test_a, test_b)
    overall = df[TARGET].value_counts(normalize=True)
    in_test = test_a[TARGET].value_counts(normalize=True)
    assert (overall - in_test).abs().max() < 0.05


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_bad_test_size(raw_df: pd.DataFrame, test_size: float) -> None:
    with pytest.raises(ValueError, match="test_size"):
        train_test_split_df(clean(raw_df), test_size=test_size)


def test_split_rejects_unknown_strategy(raw_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="strategy"):
        train_test_split_df(clean(raw_df), strategy="random")
