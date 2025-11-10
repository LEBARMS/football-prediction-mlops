# app/main.py
import os
import xgboost as xgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal

app = FastAPI(title="Football Prediction API", docs_url="/", redoc_url="/redoc")

# ---- Model paths (match your current dvc.yaml)
# You can override at runtime:  MODEL_DIR=/path/to/models
MODEL_DIR = os.getenv("MODEL_DIR", "app/model")
HOME_MODEL_PATH = os.path.join(MODEL_DIR, "home_model.json")
AWAY_MODEL_PATH = os.path.join(MODEL_DIR, "away_model.json")

# ---- Exact feature schema used by train/predict
REQUIRED_FEATURES = [
    "home_matches_played", "home_goals_for", "home_goals_against", "home_goals_diff",
    "away_matches_played", "away_goals_for", "away_goals_against", "away_goals_diff",
]

# ---------- Pydantic payloads ----------
class MatchFeatures(BaseModel):
    home_matches_played: float = Field(..., ge=0)
    home_goals_for: float = Field(..., ge=0)
    home_goals_against: float = Field(..., ge=0)
    home_goals_diff: float
    away_matches_played: float = Field(..., ge=0)
    away_goals_for: float = Field(..., ge=0)
    away_goals_against: float = Field(..., ge=0)
    away_goals_diff: float

class PredictOneRequest(MatchFeatures):
    pass

class PredictBatchRequest(BaseModel):
    items: List[MatchFeatures]

class PredictOneResponse(BaseModel):
    pred_home_goals: float
    pred_away_goals: float
    predicted_result: Literal["Home Win", "Away Win", "Draw"]

class PredictBatchResponse(BaseModel):
    predictions: List[PredictOneResponse]

# ---------- Model loading (startup-safe) ----------
home_model = xgb.XGBRegressor()
away_model = xgb.XGBRegressor()
_models_loaded = False  # we keep track to avoid crashing on startup

def _models_exist() -> bool:
    return all(os.path.exists(p) and os.path.getsize(p) > 0 for p in (HOME_MODEL_PATH, AWAY_MODEL_PATH))

def _try_load_models() -> bool:
    """Attempt to load models once; return True if successful, False otherwise."""
    global _models_loaded
    if _models_loaded:
        return True
    if not _models_exist():
        return False
    try:
        home_model.load_model(HOME_MODEL_PATH)
        away_model.load_model(AWAY_MODEL_PATH)
        _models_loaded = True
        return True
    except Exception:
        return False

@app.on_event("startup")
def _startup():
    # Best-effort load; do not crash the server if models are absent
    _try_load_models()

@app.get("/health", include_in_schema=False)
def health():
    return {
        "status": "ok" if _models_loaded else "degraded",
        "models_loaded": _models_loaded,
        "model_dir": MODEL_DIR,
        "home_model": os.path.basename(HOME_MODEL_PATH),
        "away_model": os.path.basename(AWAY_MODEL_PATH),
    }

# ---------- Core prediction ----------
def _predict_core(df: pd.DataFrame) -> List[PredictOneResponse]:
    # Ensure models are available (lazy load if needed)
    if not _try_load_models():
        raise HTTPException(status_code=503, detail=f"Models not available in {MODEL_DIR}")

    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    X = df[REQUIRED_FEATURES]
    pred_h = home_model.predict(X)
    pred_a = away_model.predict(X)

    def outcome(h, a):
        if h > a: return "Home Win"
        if h < a: return "Away Win"
        return "Draw"

    return [
        PredictOneResponse(
            pred_home_goals=float(h),
            pred_away_goals=float(a),
            predicted_result=outcome(h, a),
        )
        for h, a in zip(pred_h, pred_a)
    ]

# ---------- Endpoints ----------
@app.post("/predict_one", response_model=PredictOneResponse)
def predict_one(payload: PredictOneRequest):
    df = pd.DataFrame([payload.dict()])
    return _predict_core(df)[0]

@app.post("/predict", response_model=PredictBatchResponse)
def predict_batch(payload: PredictBatchRequest):
    df = pd.DataFrame([item.dict() for item in payload.items])
    preds = _predict_core(df)
    return PredictBatchResponse(predictions=preds)
