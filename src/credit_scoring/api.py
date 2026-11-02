"""FastAPI inference service.

Run with ``uvicorn credit_scoring.api:app`` or ``credit-scoring serve``.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from credit_scoring import __version__
from credit_scoring.config import Settings, get_settings
from credit_scoring.predict import CreditRecord, Predictor

logger = logging.getLogger(__name__)


class Prediction(BaseModel):
    """Model output for a single record."""

    credit_score: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]


class BatchRequest(BaseModel):
    """Batch of records to score."""

    records: list[CreditRecord] = Field(min_length=1)


class BatchResponse(BaseModel):
    """Predictions in the same order as the request."""

    model_name: str
    predictions: list[Prediction]


def get_predictor(request: Request) -> Predictor:
    """Dependency returning the loaded predictor, or 503 if none is loaded."""
    loaded = getattr(request.app.state, "predictor", None)
    if loaded is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Train a model first (`credit-scoring train`).",
        )
    return loaded


LoadedPredictor = Annotated[Predictor, Depends(get_predictor)]


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    """Application factory.

    Args:
        settings: Overrides :func:`get_settings` (useful in tests).
        predictor: Pre-loaded predictor; if ``None`` the artifact at
            ``settings.model_path`` is loaded on startup.
    """
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = predictor
        if predictor is None:
            try:
                app.state.predictor = Predictor.from_path(settings.model_path)
                logger.info("Loaded model from %s", settings.model_path)
            except FileNotFoundError as exc:
                # Start anyway so /health can report the problem.
                logger.error("%s", exc)
        yield

    app = FastAPI(
        title="Credit Score Classifier",
        version=__version__,
        description="Predict a customer's credit score bracket (Poor / Standard / Good).",
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        loaded = getattr(request.app.state, "predictor", None)
        return {"status": "ok", "model_loaded": loaded is not None, "version": __version__}

    @app.get("/model/info")
    def model_info(model: LoadedPredictor) -> dict[str, Any]:
        meta = model.metadata
        metrics = {
            k: v
            for k, v in meta.get("metrics", {}).items()
            if k in {"accuracy", "f1_weighted", "f1_macro", "cohen_kappa", "split_strategy"}
        }
        return {
            "model_name": model.model_name,
            "class_labels": model.class_labels,
            "trained_at": meta.get("trained_at"),
            "metrics": metrics,
        }

    @app.post("/predict", response_model=Prediction)
    def predict(record: CreditRecord, model: LoadedPredictor) -> dict[str, Any]:
        return model.predict([record])[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(batch: BatchRequest, model: LoadedPredictor) -> dict[str, Any]:
        if len(batch.records) > settings.api_max_batch_size:
            raise HTTPException(
                413,  # Content Too Large; the starlette constant was renamed across versions
                detail=f"Batch size {len(batch.records)} exceeds limit "
                f"{settings.api_max_batch_size}.",
            )
        return {"model_name": model.model_name, "predictions": model.predict(batch.records)}

    return app


app = create_app()
