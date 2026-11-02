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
RUN pip install --no-deps . \
    # Trim what the runtime never needs: bundled test suites and pip itself.
    && find /opt/venv -depth -type d -name tests -exec rm -rf {} + \
    && pip uninstall -y pip

# ---- Runtime: slim image, non-root user, no extra system packages ----
FROM python:3.12-slim AS runtime

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
# Uses Python's stdlib instead of installing curl into the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]

# The CLI configures logging, so model-loading messages appear in `docker logs`.
CMD ["credit-scoring", "serve", "--host", "0.0.0.0", "--port", "8000"]
