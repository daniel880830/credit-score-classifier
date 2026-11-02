"""Model factory: builds an end-to-end sklearn ``Pipeline`` for each algorithm."""

from __future__ import annotations

from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from credit_scoring.features import build_preprocessor

SUPPORTED_MODELS: tuple[str, ...] = ("knn", "xgboost", "random_forest")


def _make_classifier(name: str, params: dict[str, Any], random_state: int) -> ClassifierMixin:
    if name == "knn":
        return KNeighborsClassifier(**params)
    if name == "random_forest":
        params = dict(params)
        # TOML has no null; 0 means "grow trees fully".
        if params.get("max_depth") == 0:
            params["max_depth"] = None
        return RandomForestClassifier(random_state=random_state, **params)
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("xgboost is not installed: pip install xgboost") from exc
        return XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            **params,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {SUPPORTED_MODELS}")


def build_pipeline(
    name: str,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Build an unfitted ``preprocess -> classifier`` pipeline.

    Args:
        name: One of :data:`SUPPORTED_MODELS`.
        params: Classifier keyword arguments (e.g. from ``model_params.toml``).
        random_state: Seed passed to stochastic classifiers.

    Returns:
        A scikit-learn :class:`~sklearn.pipeline.Pipeline` with steps
        ``"preprocess"`` and ``"clf"``.

    Raises:
        ValueError: If ``name`` is not supported.
    """
    clf = _make_classifier(name, params or {}, random_state)
    return Pipeline(
        [
            ("preprocess", build_preprocessor(scale_numeric=name == "knn")),
            ("clf", clf),
        ]
    )
