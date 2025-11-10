# runtime-only image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# (1) OS deps needed by xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# (2) Install only runtime deps
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# (3) Copy only what the API needs
COPY app/ app/
COPY app/model ./app/model

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
