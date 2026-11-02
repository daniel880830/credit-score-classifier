from pathlib import Path

import pytest
from pydantic import ValidationError

from credit_scoring.config import PROJECT_ROOT, Settings, load_model_params


def test_defaults_are_relative_to_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CREDIT_DATA_PATH", raising=False)
    s = Settings(_env_file=None)
    assert s.data_path.is_relative_to(PROJECT_ROOT)
    assert s.model_path == s.model_dir / "xgboost.joblib"


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CREDIT_DATA_PATH", str(tmp_path / "x.csv"))
    monkeypatch.setenv("CREDIT_MODEL_NAME", "knn")
    monkeypatch.setenv("CREDIT_TEST_SIZE", "0.3")
    s = Settings(_env_file=None)
    assert s.data_path == tmp_path / "x.csv"
    assert s.model_path.name == "knn.joblib"
    assert s.test_size == 0.3


def test_dotenv_file_is_read(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("CREDIT_SPLIT_STRATEGY=row\nCREDIT_RANDOM_STATE=7\n")
    s = Settings(_env_file=env)
    assert (s.split_strategy, s.random_state) == ("row", 7)


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("CREDIT_MODEL_NAME", "svm"),
        ("CREDIT_SPLIT_STRATEGY", "random"),
        ("CREDIT_TEST_SIZE", "1.2"),
        ("CREDIT_API_MAX_BATCH_SIZE", "0"),
    ],
)
def test_invalid_env_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch, var: str, value: str
) -> None:
    monkeypatch.setenv(var, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_load_model_params_reads_repo_config() -> None:
    params = load_model_params(PROJECT_ROOT / "config" / "model_params.toml")
    assert {"knn", "xgboost", "random_forest"} <= params.keys()
    assert params["knn"]["params"]["metric"] == "l1"


def test_load_model_params_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model_params(tmp_path / "missing.toml")
