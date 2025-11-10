FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/model

WORKDIR /app

# xgboost runtime dep
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# install API deps
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# copy app and models (CI should have run `dvc pull` before building)
COPY app/ ./app
COPY app/model ./app/model

# add entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
CMD ["/entrypoint.sh"]
