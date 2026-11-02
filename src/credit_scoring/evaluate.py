"""Model evaluation metrics and inference benchmarks."""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from credit_scoring.schema import CLASS_LABELS

ArrayLike = np.ndarray | pd.Series


def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, Any]:
    """Compute classification metrics for integer-encoded labels.

    Args:
        y_true: Ground-truth class ids.
        y_pred: Predicted class ids.

    Returns:
        Accuracy, weighted/macro F1, Cohen's kappa, the confusion matrix and a
        per-class report keyed by label name.

    Raises:
        ValueError: If inputs are empty or have different lengths.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on empty inputs.")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    labels = list(range(len(CLASS_LABELS)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(CLASS_LABELS),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {label: report[label] for label in CLASS_LABELS},
    }


def benchmark_inference(
    pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    n_single: int = 200,
    batch_size: int = 1000,
) -> dict[str, float]:
    """Measure single-row latency, batch throughput and batch peak memory.

    Args:
        pipeline: A fitted pipeline.
        X: Feature rows to sample from (at least one row).
        n_single: Number of single-row predictions to time.
        batch_size: Rows in the throughput/memory batch.

    Returns:
        ``latency_p50_ms``, ``latency_p95_ms``, ``throughput_rows_per_s`` and
        ``batch_peak_memory_mb``.
    """
    if X.empty:
        raise ValueError("Need at least one row to benchmark.")
    n_single = min(n_single, len(X))
    timings = []
    for i in range(n_single):
        row = X.iloc[[i]]
        start = time.perf_counter()
        pipeline.predict(row)
        timings.append((time.perf_counter() - start) * 1000)

    batch = X.iloc[: min(batch_size, len(X))]
    tracemalloc.start()
    start = time.perf_counter()
    pipeline.predict(batch)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "latency_p50_ms": float(np.percentile(timings, 50)),
        "latency_p95_ms": float(np.percentile(timings, 95)),
        "throughput_rows_per_s": float(len(batch) / elapsed) if elapsed > 0 else float("inf"),
        "batch_peak_memory_mb": peak / 1024**2,
    }


def file_size_mb(path: Path) -> float:
    """Size of ``path`` on disk in megabytes."""
    return Path(path).stat().st_size / 1024**2


def plot_confusion_matrix(matrix: list[list[int]], title: str, out_path: Path) -> Path:
    """Save a confusion-matrix heatmap as PNG (requires matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(CLASS_LABELS)), CLASS_LABELS)
    ax.set_yticks(range(len(CLASS_LABELS)), CLASS_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for (i, j), value in np.ndenumerate(arr):
        color = "white" if value > arr.max() / 2 else "black"
        ax.text(j, i, str(value), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
