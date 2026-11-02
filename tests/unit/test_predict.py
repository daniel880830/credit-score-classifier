from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from credit_scoring.predict import CreditRecord, Predictor, records_to_frame
from credit_scoring.schema import FEATURE_COLS, LOAN_TYPE_COLS


def test_valid_record_parses(valid_payload: dict) -> None:
    record = CreditRecord(**valid_payload)
    assert record.loan_types == ["Auto Loan", "Student Loan"]  # "and " prefix stripped


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("annual_income", 0),
        ("annual_income", -1),
        ("record_month", 13),
        ("record_month", 0),
        ("customer_age", 5),
        ("customer_age", 150),
        ("credit_utilization", 101),
        ("credit_mix", "Excellent"),
        ("payment_behavior", "Unknown"),
        ("customer_occupation", ""),
        ("loan_types", ["Auto Loan"] * 10),
        ("loan_types", ["  "]),
        ("outstanding_debt", "lots"),
        ("annual_income", None),
    ],
)
def test_invalid_field_values_rejected(valid_payload: dict, field: str, value: object) -> None:
    valid_payload[field] = value
    with pytest.raises(ValidationError):
        CreditRecord(**valid_payload)


def test_missing_field_rejected(valid_payload: dict) -> None:
    del valid_payload["credit_mix"]
    with pytest.raises(ValidationError, match="credit_mix"):
        CreditRecord(**valid_payload)


def test_extra_field_rejected(valid_payload: dict) -> None:
    valid_payload["ssn"] = "123"
    with pytest.raises(ValidationError, match="ssn"):
        CreditRecord(**valid_payload)


def test_loan_types_defaults_to_empty(valid_payload: dict) -> None:
    del valid_payload["loan_types"]
    frame = records_to_frame([CreditRecord(**valid_payload)])
    assert frame[LOAN_TYPE_COLS].isna().all(axis=None)


def test_records_to_frame_shape_and_padding(valid_payload: dict) -> None:
    frame = records_to_frame([CreditRecord(**valid_payload)] * 3)
    assert list(frame.columns) == FEATURE_COLS
    assert len(frame) == 3
    assert frame.loc[0, "loan_types"] == "Auto Loan"
    assert frame.loc[0, "loan_types_2"] == "Student Loan"
    assert np.isnan(frame.loc[0, "loan_types_3"])
    assert frame.loc[0, "min_amount_payment"] == 0


def test_records_to_frame_rejects_empty() -> None:
    with pytest.raises(ValueError, match="At least one"):
        records_to_frame([])


def test_predictor_rejects_invalid_artifact() -> None:
    with pytest.raises(ValueError, match="pipeline"):
        Predictor({"model_name": "x", "class_labels": []})


def test_predictor_from_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="credit-scoring train"):
        Predictor.from_path(tmp_path / "missing.joblib")


def test_predictor_from_corrupt_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a pickle")
    with pytest.raises(Exception):  # noqa: B017 - joblib raises various unpickling errors
        Predictor.from_path(bad)
