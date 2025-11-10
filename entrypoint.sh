#!/bin/sh
set -eu

echo "== Boot info =="
echo "PWD: $(pwd)"
echo "Python: $(python --version)"
echo "PORT: ${PORT:-8080}"
echo "MODEL_DIR: ${MODEL_DIR:-/app/model}"
echo "App tree:"
ls -lah /app || true
ls -lah /app/app || true
ls -lah /app/model || true

echo "== Import sanity =="
python - <<'PY'
import sys, os, importlib
mods = ["uvicorn", "fastapi", "xgboost", "pandas", "pydantic"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"OK import: {m}")
    except Exception as e:
        print(f"FAIL import: {m} -> {e}")
        sys.exit(2)
print("All critical imports OK")
PY

echo "== Starting Uvicorn =="
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}" --log-level debug
