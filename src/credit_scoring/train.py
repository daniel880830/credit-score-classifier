"""Training, hyperparameter tuning and artifact persistence."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from credit_scoring import __version__
from credit_scoring.config import Settings, load_model_params
from credit_scoring.data import clean, load_raw, split_features_target, train_test_split_df
from credit_scoring.evaluate import (
    benchmark_inference,
    compute_metrics,
    file_size_mb,
    plot_confusion_matrix,
)
from credit_scoring.models import build_pipeline
from credit_scoring.schema import CLASS_LABELS, FEATURE_COLS, GROUP_COL

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Outcome of a single training run."""

    model_name: str
    model_path: Path
    metrics: dict[str, Any]
    benchmark: dict[str, float] = field(default_factory=dict)


def prepare_data(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean and split the dataset according to ``settings``."""
    df = clean(load_raw(settings.data_path))
    return train_test_split_df(
        df,
        strategy=settings.split_strategy,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )


def configure_for_serving(pipeline: Pipeline) -> Pipeline:
    """Switch a fitted classifier to single-threaded prediction.

    Training uses every core (``n_jobs=-1``), but the API mostly scores one
    record at a time, where spinning up a thread pool costs more than the
    prediction itself (~25 ms for the random forest).
    """
    clf = pipeline.named_steps["clf"]
    if "n_jobs" in clf.get_params():
        clf.set_params(n_jobs=1)
    return pipeline


def save_artifact(pipeline: Pipeline, model_name: str, metrics: dict[str, Any], path: Path) -> Path:
    """Persist a fitted pipeline together with the metadata the API needs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "model_name": model_name,
        "class_labels": list(CLASS_LABELS),
        "feature_cols": list(FEATURE_COLS),
        "metrics": metrics,
        "package_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    joblib.dump(artifact, path, compress=3)
    return path


def train_model(
    settings: Settings,
    model_name: str,
    data: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    *,
    benchmark: bool = True,
) -> TrainResult:
    """Fit ``model_name`` on the train split, evaluate on test and save artifacts.

    Args:
        settings: Runtime settings (paths, split, seed).
        model_name: One of ``knn``, ``xgboost``, ``random_forest``.
        data: Optional pre-computed ``(train_df, test_df)`` to avoid re-cleaning.
        benchmark: Also measure inference latency / memory.

    Returns:
        A :class:`TrainResult` with metrics and the artifact path.
    """
    train_df, test_df = data if data is not None else prepare_data(settings)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    params = load_model_params(settings.params_path).get(model_name, {}).get("params", {})
    pipeline = build_pipeline(model_name, params, random_state=settings.random_state)

    logger.info("Training %s on %d rows", model_name, len(X_train))
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    configure_for_serving(pipeline)

    metrics = compute_metrics(y_test, pipeline.predict(X_test))
    metrics.update(
        {
            "fit_seconds": fit_seconds,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "split_strategy": settings.split_strategy,
        }
    )

    model_path = settings.model_dir / f"{model_name}.joblib"
    save_artifact(pipeline, model_name, metrics, model_path)

    bench: dict[str, float] = {}
    if benchmark:
        bench = benchmark_inference(pipeline, X_test)
        bench["model_size_mb"] = file_size_mb(model_path)

    reports = settings.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"metrics_{model_name}.json").write_text(
        json.dumps({"metrics": metrics, "benchmark": bench}, indent=2), encoding="utf-8"
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        f"{model_name} ({settings.split_strategy} split)",
        reports / "figures" / f"confusion_{model_name}.png",
    )
    logger.info(
        "%s: acc=%.4f f1_w=%.4f kappa=%.4f",
        model_name,
        metrics["accuracy"],
        metrics["f1_weighted"],
        metrics["cohen_kappa"],
    )
    return TrainResult(model_name, model_path, metrics, bench)


def tune_model(settings: Settings, model_name: str, n_splits: int = 3) -> dict[str, Any]:
    """Grid-search the ``[<model>.grid]`` space with customer-grouped CV.

    Returns:
        ``best_params`` and ``best_score`` (weighted F1).
    """
    train_df, _ = prepare_data(settings)
    X, y = split_features_target(train_df)
    config = load_model_params(settings.params_path).get(model_name, {})
    grid = config.get("grid")
    if not grid:
        raise ValueError(f"No [{model_name}.grid] section in {settings.params_path}")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=settings.random_state)
    search = GridSearchCV(
        build_pipeline(model_name, config.get("params", {}), settings.random_state),
        grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y, groups=train_df[GROUP_COL])
    return {"best_params": search.best_params_, "best_score": float(search.best_score_)}
