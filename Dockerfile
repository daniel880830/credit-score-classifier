# syntax=docker/dockerfile:1

# ---- Builder: resolve and install dependencies into an isolated venv ----
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
# Copy only dependency manifests first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# ---- Runtime: slim image, non-root user ----
FROM python:3.12-slim AS runtime

# libgomp1 is required by XGBoost's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CREDIT_DATA_PATH=/app/data/raw/credit_records.csv \
    CREDIT_MODEL_DIR=/app/models \
    CREDIT_REPORTS_DIR=/app/reports \
    CREDIT_PARAMS_PATH=/app/config/model_params.toml

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
RUN useradd --create-home --uid 1000 appuser
COPY --chown=appuser:appuser config ./config
COPY --chown=appuser:appuser data ./data
RUN mkdir -p models reports && chown appuser:appuser models reports
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "credit_scoring.api:app", "--host", "0.0.0.0", "--port", "8000"]
