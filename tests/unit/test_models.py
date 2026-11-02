import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from credit_scoring.data import clean, split_features_target
from credit_scoring.models import SUPPORTED_MODELS, build_pipeline


@pytest.fixture
def Xy(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return split_features_target(clean(raw_df))


@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_build_pipeline_fits_and_predicts_valid_classes(
    name: str, Xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, y = Xy
    params = {"n_estimators": 10} if name != "knn" else {"n_neighbors": 3}
    pipe = build_pipeline(name, params).fit(X, y)
    pred = pipe.predict(X.iloc[:10])
    assert set(pred) <= {0, 1, 2}
    proba = pipe.predict_proba(X.iloc[:10])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


def test_build_pipeline_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("svm")


def test_build_pipeline_bad_param_raises() -> None:
    with pytest.raises(TypeError):
        build_pipeline("knn", {"not_a_param": 1})


def test_random_forest_zero_depth_means_unlimited() -> None:
    clf = build_pipeline("random_forest", {"max_depth": 0})["clf"]
    assert isinstance(clf, RandomForestClassifier)
    assert clf.max_depth is None


def test_only_knn_scales_numeric_features() -> None:
    knn_numeric = build_pipeline("knn")["preprocess"].transformers[0][1]
    rf_numeric = build_pipeline("random_forest")["preprocess"].transformers[0][1]
    assert any(isinstance(s, StandardScaler) for s in knn_numeric.named_steps.values())
    assert not any(isinstance(s, StandardScaler) for s in rf_numeric.named_steps.values())


def test_knn_nearest_neighbour_retrieval_returns_exact_match(
    Xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    """With distance weighting, a query identical to a stored vector gets its label."""
    X, y = Xy
    pipe = build_pipeline("knn", {"n_neighbors": 5, "weights": "distance"}).fit(X, y)
    assert isinstance(pipe["clf"], KNeighborsClassifier)
    np.testing.assert_array_equal(pipe.predict(X.iloc[:25]), y.iloc[:25].to_numpy())


def test_xgboost_respects_random_state(Xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = Xy
    params = {"n_estimators": 10, "subsample": 0.7, "colsample_bytree": 0.7}
    a = build_pipeline("xgboost", params, random_state=7).fit(X, y).predict_proba(X)
    b = build_pipeline("xgboost", params, random_state=7).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b)
