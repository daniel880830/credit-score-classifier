"""Inference: request schema and a predictor that wraps a saved artifact."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from credit_scoring.schema import FEATURE_COLS, LOAN_TYPE_COLS, PAYMENT_BEHAVIORS

PaymentBehavior = Literal[PAYMENT_BEHAVIORS]  # type: ignore[valid-type]


class CreditRecord(BaseModel):
    """One monthly customer record to score."""

    model_config = ConfigDict(extra="forbid")

    record_month: int = Field(ge=1, le=12)
    customer_age: int = Field(ge=14, le=100)
    customer_occupation: str = Field(min_length=1, max_length=64)
    annual_income: float = Field(gt=0)
    monthly_salary: float = Field(ge=0)
    bank_accounts: float = Field(ge=0)
    credit_cards: int = Field(ge=0)
    interest_rate: float = Field(ge=0)
    loans: int = Field(ge=0)
    loan_types: list[str] = Field(default_factory=list, max_length=len(LOAN_TYPE_COLS))
    payment_delay_days: int
    delayed_payments: float = Field(ge=0)
    credit_limit_changes: float
    credit_inquiries: float = Field(ge=0)
    credit_mix: Literal["Bad", "Standard", "Good"]
    outstanding_debt: float = Field(ge=0)
    credit_utilization: float = Field(ge=0, le=100)
    credit_history_age_months: float = Field(ge=0)
    min_amount_payment: bool
    monthly_emi: float = Field(ge=0)
    monthly_investment: float = Field(ge=0)
    payment_behavior: PaymentBehavior
    monthly_balance: float

    @field_validator("loan_types")
    @classmethod
    def _strip_loan_types(cls, value: list[str]) -> list[str]:
        cleaned = [v.strip().removeprefix("and ").strip() for v in value]
        if any(not v for v in cleaned):
            raise ValueError("loan_types entries must be non-empty strings")
        return cleaned


def records_to_frame(records: Sequence[CreditRecord]) -> pd.DataFrame:
    """Convert validated records into the wide frame the pipeline expects.

    ``loan_types`` (a list) is spread across the ``loan_types``..``loan_types_9``
    slot columns, padding with ``NaN``.

    Raises:
        ValueError: If ``records`` is empty.
    """
    if not records:
        raise ValueError("At least one record is required.")
    rows: list[dict[str, Any]] = []
    for record in records:
        row = record.model_dump()
        loans = row.pop("loan_types")
        row["min_amount_payment"] = int(row["min_amount_payment"])
        for i, col in enumerate(LOAN_TYPE_COLS):
            row[col] = loans[i] if i < len(loans) else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURE_COLS)


class Predictor:
    """Thin wrapper around a saved artifact that returns labelled predictions."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        missing = {"pipeline", "class_labels", "model_name"} - artifact.keys()
        if missing:
            raise ValueError(f"Invalid model artifact, missing keys: {sorted(missing)}")
        self.pipeline = artifact["pipeline"]
        self.class_labels: list[str] = list(artifact["class_labels"])
        self.model_name: str = artifact["model_name"]
        self.metadata: dict[str, Any] = {k: v for k, v in artifact.items() if k not in {"pipeline"}}

    @classmethod
    def from_path(cls, path: Path) -> Predictor:
        """Load an artifact written by :func:`credit_scoring.train.save_artifact`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: {path}. Run `credit-scoring train`."
            )
        return cls(joblib.load(path))

    def predict(self, records: Sequence[CreditRecord]) -> list[dict[str, Any]]:
        """Score records and return the label plus per-class probabilities."""
        X = records_to_frame(records)
        proba = self.pipeline.predict_proba(X)
        results = []
        for row in proba:
            idx = int(np.argmax(row))
            results.append(
                {
                    "credit_score": self.class_labels[idx],
                    "confidence": float(row[idx]),
                    "probabilities": {
                        label: float(p) for label, p in zip(self.class_labels, row, strict=True)
                    },
                }
            )
        return results
