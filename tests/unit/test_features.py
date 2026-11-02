import numpy as np
import pandas as pd
import pytest

from credit_scoring.data import clean, split_features_target
from credit_scoring.features import MultiHotEncoder, build_preprocessor
from credit_scoring.schema import LOAN_TYPE_COLS

# ------------------------------------------------------------- MultiHotEncoder


@pytest.fixture
def loans() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": ["Auto Loan", "Payday Loan", np.nan],
            "b": ["Payday Loan", np.nan, np.nan],
        }
    )


def test_multihot_learns_sorted_vocab_ignoring_nan(loans: pd.DataFrame) -> None:
    enc = MultiHotEncoder().fit(loans)
    assert enc.categories_ == ["Auto Loan", "Payday Loan"]


def test_multihot_transform_marks_presence_once(loans: pd.DataFrame) -> None:
    out = MultiHotEncoder().fit(loans).transform(loans)
    np.testing.assert_array_equal(out, [[1, 1], [0, 1], [0, 0]])


def test_multihot_single_row_keeps_training_width(loans: pd.DataFrame) -> None:
    enc = MultiHotEncoder().fit(loans)
    out = enc.transform(loans.iloc[[2]])  # all-NaN row
    assert out.shape == (1, 2)
    assert out.sum() == 0


def test_multihot_ignores_unknown_values(loans: pd.DataFrame) -> None:
    enc = MultiHotEncoder().fit(loans)
    unseen = pd.DataFrame({"a": ["Space Loan"], "b": [np.nan]})
    assert enc.transform(unseen).sum() == 0


def test_multihot_feature_names(loans: pd.DataFrame) -> None:
    names = MultiHotEncoder().fit(loans).get_feature_names_out()
    assert list(names) == ["loan_Auto Loan", "loan_Payday Loan"]


def test_multihot_all_missing_column_gives_empty_vocab() -> None:
    df = pd.DataFrame({"a": [np.nan, np.nan]})
    enc = MultiHotEncoder().fit(df)
    assert enc.transform(df).shape == (2, 0)


# --------------------------------------------------------------- preprocessor


@pytest.fixture
def X(raw_df: pd.DataFrame) -> pd.DataFrame:
    return split_features_target(clean(raw_df))[0]


def test_preprocessor_output_is_numeric_and_finite(X: pd.DataFrame) -> None:
    out = build_preprocessor().fit_transform(X)
    assert np.isfinite(np.asarray(out, dtype=float)).all()


def test_preprocessor_single_row_matches_training_width(X: pd.DataFrame) -> None:
    pre = build_preprocessor().fit(X)
    assert pre.transform(X.iloc[[0]]).shape[1] == pre.transform(X).shape[1]


def test_preprocessor_handles_unseen_categories(X: pd.DataFrame) -> None:
    pre = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row["customer_occupation"] = "Astronaut"
    row["credit_mix"] = "Excellent"
    row[LOAN_TYPE_COLS[0]] = "Space Loan"
    out = pre.transform(row)
    assert out.shape[1] == pre.transform(X.iloc[[0]]).shape[1]
    names = list(pre.get_feature_names_out())
    assert out[0, names.index("credit_mix")] == -1


def test_preprocessor_imputes_missing_numeric(X: pd.DataFrame) -> None:
    pre = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row["monthly_balance"] = np.nan
    assert np.isfinite(pre.transform(row).astype(float)).all()


def test_preprocessor_scaling_toggle(X: pd.DataFrame) -> None:
    scaled = build_preprocessor(scale_numeric=True).fit_transform(X)
    raw = build_preprocessor(scale_numeric=False).fit_transform(X)
    # annual_income is column 2 of the numeric block.
    assert abs(scaled[:, 2].mean()) < 1e-6
    assert raw[:, 2].mean() > 1000
