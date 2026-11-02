"""Feature engineering: scikit-learn transformers used inside every model pipeline."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from credit_scoring.schema import CREDIT_MIX_ORDER, LOAN_TYPE_COLS, NOMINAL_COLS, NUMERIC_COLS


class MultiHotEncoder(TransformerMixin, BaseEstimator):
    """Encode several "slot" columns holding the same vocabulary as one multi-hot block.

    A customer's loans are spread over ``loan_types`` ... ``loan_types_9``;
    the order is meaningless, so we emit one 0/1 column per loan type.
    Unlike ``pd.get_dummies`` the vocabulary is learned in :meth:`fit`, so a
    single-row request at inference time yields the same columns as training.
    Unknown values are ignored.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Learn the sorted vocabulary of non-null values across all columns."""
        values = pd.unique(np.asarray(X, dtype=object).ravel())
        self.categories_: list[str] = sorted(str(v) for v in values if not pd.isna(v))
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Return an ``(n_samples, n_categories)`` 0/1 matrix."""
        arr = np.asarray(X, dtype=object)
        out = np.zeros((arr.shape[0], len(self.categories_)), dtype=np.int8)
        for j, category in enumerate(self.categories_):
            out[:, j] = (arr == category).any(axis=1)
        return out

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Feature names such as ``loan_Auto Loan``."""
        return np.asarray([f"loan_{c}" for c in self.categories_], dtype=object)


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Create the column-wise preprocessing step.

    Args:
        scale_numeric: Standardise numeric features. Distance-based models
            (KNN) need it; tree models are scale-invariant.

    Returns:
        An unfitted :class:`~sklearn.compose.ColumnTransformer`.
    """
    numeric_steps: list[object] = [SimpleImputer(strategy="median")]
    if scale_numeric:
        numeric_steps.append(StandardScaler())

    return ColumnTransformer(
        [
            ("numeric", make_pipeline(*numeric_steps), NUMERIC_COLS),
            (
                "credit_mix",
                OrdinalEncoder(
                    categories=[CREDIT_MIX_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ["credit_mix"],
            ),
            ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL_COLS),
            ("loans", MultiHotEncoder(), LOAN_TYPE_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
