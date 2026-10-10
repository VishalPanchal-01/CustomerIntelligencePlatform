# ============================================================
# BASE IMAGE
# ============================================================

FROM python:3.12-slim


# ============================================================
# PYTHON ENVIRONMENT
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1


# ============================================================
# WORKING DIRECTORY
# ============================================================

WORKDIR /app


# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# PYTHON DEPENDENCIES
# ============================================================

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt


# ============================================================
# API APPLICATION
# ============================================================

COPY api/ ./api/


# ============================================================
# SOURCE MODULES
# ============================================================

COPY src/ ./src/


# ============================================================
# PERSISTED MODELS
# ============================================================

COPY models/ ./models/


# ============================================================
# DASHBOARD CUSTOMER INTELLIGENCE ARTIFACT
# ============================================================

COPY artifacts/dashboard/ \
    ./artifacts/dashboard/


# ============================================================
# RECOMMENDATION PREDICTIONS
# ============================================================

COPY artifacts/recommendation/predictions/ \
    ./artifacts/recommendation/predictions/


# ============================================================
# SHAP EXPLAINABILITY ARTIFACTS
# ============================================================

COPY artifacts/explainability/ \
    ./artifacts/explainability/


# ============================================================
# NON-ROOT USER
# ============================================================

RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup appuser \
    && chown -R appuser:appgroup /app

USER appuser


# ============================================================
# PORT
# ============================================================

EXPOSE 8000


# ============================================================
# HEALTH CHECK
# ============================================================

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=30s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" \
    || exit 1


# ============================================================
# START API
# ============================================================

CMD [
    "uvicorn",
    "api.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]