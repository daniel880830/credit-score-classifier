import json

import numpy as np
import pytest

from credit_scoring import cli
from credit_scoring.config import PROJECT_ROOT, Settings
from credit_scoring.data import clean, load_raw, train_test_split_df
from credit_scoring.models import SUPPORTED_MODELS
from credit_scoring.predict import CreditRecord, Predictor
from credit_scoring.train import prepare_data, train_model, tune_model

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_train_model_end_to_end(settings: Settings, name: str) -> None:
    result = train_model(settings, name)

    assert result.model_path.is_file()
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    # Synthetic labels are a simple function of the features; every model should beat chance.
    assert result.metrics["f1_weighted"] > 0.5
    assert result.benchmark["model_size_mb"] > 0

    report = json.loads((settings.reports_dir / f"metrics_{name}.json").read_text())
    assert report["metrics"]["split_strategy"] == "group"
    assert (settings.reports_dir / "figures" / f"confusion_{name}.png").is_file()


def test_saved_artifact_round_trip(settings: Settings, valid_payload: dict) -> None:
    result = train_model(settings, "random_forest", benchmark=False)
    predictor = Predictor.from_path(result.model_path)

    out = predictor.predict([CreditRecord(**valid_payload)] * 2)
    assert len(out) == 2
    assert out[0]["credit_score"] in {"Poor", "Standard", "Good"}
    assert np.isclose(sum(out[0]["probabilities"].values()), 1.0)
    assert out[0]["confidence"] == max(out[0]["probabilities"].values())
    assert predictor.metadata["metrics"]["n_train"] > 0
    assert predictor.pipeline.named_steps["clf"].n_jobs == 1  # tuned for serving


def test_train_reuses_prepared_data(settings: Settings) -> None:
    data = prepare_data(settings)
    a = train_model(settings, "knn", data, benchmark=False)
    b = train_model(settings, "knn", data, benchmark=False)
    assert a.metrics["accuracy"] == b.metrics["accuracy"]
    assert a.metrics["n_test"] == len(data[1])


def test_tune_model_returns_best_params(settings: Settings) -> None:
    result = tune_model(settings, "knn", n_splits=2)
    assert result["best_params"]["clf__n_neighbors"] in {3, 5}
    assert 0.0 <= result["best_score"] <= 1.0


def test_tune_model_without_grid_raises(settings: Settings) -> None:
    with pytest.raises(ValueError, match="grid"):
        tune_model(settings, "random_forest")


def test_cli_train(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    assert cli.main(["train", "--model", "knn", "--split", "row", "--no-benchmark"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert set(summary["knn"]) == {"accuracy", "f1_weighted", "cohen_kappa"}


def test_cli_rejects_unknown_model() -> None:
    with pytest.raises(SystemExit):
        cli.main(["train", "--model", "svm"])


@pytest.mark.slow
def test_real_dataset_cleans_to_expected_size() -> None:
    path = PROJECT_ROOT / "data" / "raw" / "credit_records.csv"
    if not path.is_file():
        pytest.skip("real dataset not present")
    df = clean(load_raw(path))
    # Same row count as the original notebook's 80/20 split (66,819 + 16,705).
    assert len(df) == 83_524
    assert df["credit_history_age_months"].between(1, 500).all()
    train, test = train_test_split_df(df, strategy="group")
    assert set(train["customer_id"]).isdisjoint(test["customer_id"])
