from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_scoring.data import clean, split_features_target
from credit_scoring.evaluate import (
    benchmark_inference,
    compute_metrics,
    file_size_mb,
    plot_confusion_matrix,
)
from credit_scoring.models import build_pipeline


def test_perfect_predictions() -> None:
    y = np.array([0, 1, 2, 1])
    m = compute_metrics(y, y)
    assert m["accuracy"] == m["f1_weighted"] == m["cohen_kappa"] == 1.0
    assert m["confusion_matrix"] == [[1, 0, 0], [0, 2, 0], [0, 0, 1]]


def test_confusion_matrix_is_3x3_even_if_a_class_is_absent() -> None:
    m = compute_metrics([1, 1, 1], [1, 1, 0])
    assert np.asarray(m["confusion_matrix"]).shape == (3, 3)
    assert m["per_class"]["Good"]["support"] == 0


def test_metrics_accept_pandas_series() -> None:
    m = compute_metrics(pd.Series([0, 1]), pd.Series([0, 0]))
    assert m["accuracy"] == 0.5


def test_metrics_reject_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_metrics([], [])


def test_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_metrics([0, 1, 2], [0, 1])


def test_benchmark_inference_reports_all_keys(raw_df: pd.DataFrame) -> None:
    X, y = split_features_target(clean(raw_df))
    pipe = build_pipeline("random_forest", {"n_estimators": 5}).fit(X, y)
    bench = benchmark_inference(pipe, X, n_single=5, batch_size=50)
    assert set(bench) == {
        "latency_p50_ms",
        "latency_p95_ms",
        "throughput_rows_per_s",
        "batch_peak_memory_mb",
    }
    assert 0 < bench["latency_p50_ms"] <= bench["latency_p95_ms"]
    assert bench["throughput_rows_per_s"] > 0


def test_benchmark_inference_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        benchmark_inference(build_pipeline("knn"), pd.DataFrame())


def test_plot_confusion_matrix_writes_png(tmp_path: Path) -> None:
    out = plot_confusion_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t", tmp_path / "f" / "cm.png")
    assert out.is_file()
    assert out.read_bytes()[:4] == b"\x89PNG"
    assert file_size_mb(out) > 0
