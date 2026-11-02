"""Runtime configuration.

All paths and tunables are read from environment variables (prefix ``CREDIT_``)
or a local ``.env`` file, so nothing machine-specific is hardcoded in the code.
Model hyperparameters live in a TOML file (``config/model_params.toml`` by default).
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ModelName = Literal["knn", "xgboost", "random_forest"]
SplitStrategy = Literal["group", "row"]


class Settings(BaseSettings):
    """Application settings resolved from env vars / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="CREDIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    data_path: Path = PROJECT_ROOT / "data" / "raw" / "credit_records.csv"
    model_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    params_path: Path = PROJECT_ROOT / "config" / "model_params.toml"

    model_name: ModelName = "xgboost"
    split_strategy: SplitStrategy = "group"
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = 108

    api_max_batch_size: int = Field(default=1000, gt=0)
    log_level: str = "INFO"

    @property
    def model_path(self) -> Path:
        """Location of the serialized model artifact for ``model_name``."""
        return self.model_dir / f"{self.model_name}.joblib"


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()


def load_model_params(path: Path) -> dict[str, Any]:
    """Load the hyperparameter TOML file.

    Args:
        path: Path to the TOML file.

    Returns:
        Parsed TOML as a nested dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Model params file not found: {path}")
    with path.open("rb") as fh:
        return tomllib.load(fh)
