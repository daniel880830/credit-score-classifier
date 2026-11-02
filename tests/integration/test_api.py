from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from credit_scoring.api import create_app
from credit_scoring.config import Settings
from credit_scoring.predict import Predictor
from credit_scoring.train import train_model

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def predictor(settings: Settings) -> Predictor:
    result = train_model(settings, settings.model_name, benchmark=False)
    return Predictor.from_path(result.model_path)


@pytest.fixture
def client(settings: Settings, predictor: Predictor) -> Iterator[TestClient]:
    with TestClient(create_app(settings, predictor)) as c:
        yield c


@pytest.fixture
def empty_client(settings: Settings, tmp_path) -> Iterator[TestClient]:
    no_model = settings.model_copy(update={"model_dir": tmp_path})
    with TestClient(create_app(no_model)) as c:
        yield c


def test_root_redirects_to_docs(client: TestClient) -> None:
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/docs"
    assert client.get("/").status_code == 200  # docs page itself


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client: TestClient) -> None:
    body = client.get("/model/info").json()
    assert body["model_name"] == "random_forest"
    assert body["class_labels"] == ["Poor", "Standard", "Good"]
    assert {"accuracy", "f1_weighted", "cohen_kappa"} <= body["metrics"].keys()


def test_predict_single(client: TestClient, valid_payload: dict) -> None:
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["credit_score"] in body["probabilities"]
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-6


def test_predict_unknown_occupation_and_loan_is_accepted(
    client: TestClient, valid_payload: dict
) -> None:
    valid_payload["customer_occupation"] = "Astronaut"
    valid_payload["loan_types"] = ["Space Loan"]
    assert client.post("/predict", json=valid_payload).status_code == 200


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(annual_income=-5),
        lambda p: p.update(credit_mix="Excellent"),
        lambda p: p.pop("payment_behavior"),
        lambda p: p.update(unexpected="x"),
    ],
    ids=["negative_income", "bad_enum", "missing_field", "extra_field"],
)
def test_predict_validation_errors(client: TestClient, valid_payload: dict, mutate) -> None:
    mutate(valid_payload)
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_predict_malformed_json(client: TestClient) -> None:
    resp = client.post(
        "/predict", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 422


def test_predict_batch_preserves_order(client: TestClient, valid_payload: dict) -> None:
    risky = {**valid_payload, "outstanding_debt": 4900.0, "payment_delay_days": 59}
    safe = {**valid_payload, "outstanding_debt": 10.0, "payment_delay_days": 0}
    resp = client.post("/predict/batch", json={"records": [risky, safe, risky]})
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 3
    assert preds[0] == preds[2]


def test_predict_batch_empty_rejected(client: TestClient) -> None:
    assert client.post("/predict/batch", json={"records": []}).status_code == 422


def test_predict_batch_too_large(client: TestClient, valid_payload: dict) -> None:
    resp = client.post("/predict/batch", json={"records": [valid_payload] * 6})
    assert resp.status_code == 413
    assert "exceeds limit 5" in resp.json()["detail"]


def test_no_model_returns_503(empty_client: TestClient, valid_payload: dict) -> None:
    assert empty_client.get("/health").json()["model_loaded"] is False
    assert empty_client.post("/predict", json=valid_payload).status_code == 503
    assert empty_client.get("/model/info").status_code == 503


def test_app_loads_artifact_from_disk_on_startup(
    settings: Settings, predictor: Predictor, valid_payload: dict
) -> None:
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").json()["model_loaded"] is True
        assert c.post("/predict", json=valid_payload).status_code == 200
