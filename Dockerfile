# runtime-only image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/model

WORKDIR /app

# (1) OS deps needed by xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# (2) Install only runtime deps
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# (3) Copy only what the API needs
COPY app/ ./app
# Make sure models are inside the image (CI does `dvc pull` before build)
COPY app/model ./app/model

# Cloud Run injects PORT. Don’t hardcode 8000.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
