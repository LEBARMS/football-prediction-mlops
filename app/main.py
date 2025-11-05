# app/main.py
import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Football Prediction API", docs_url="/", redoc_url="/redoc")

# Paths – match how your DVC/steps currently write and read
MODEL_DIR = "app/model"          # trained models live here
HOME_MODEL = os.path.join(MODEL_DIR, "home_model.json")
AWAY_MODEL = os.path.join(MODEL_DIR, "away_model.json")

def load_xgb(path: str) -> xgb.XGBRegressor:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"Missing model file: {path}")
    m = xgb.XGBRegressor()
    m.load_model(path)
    return m

home_model = load_xgb(HOME_MODEL)
away_model = load_xgb(AWAY_MODEL)

# Use booster feature names (the one-hot columns seen during training)
FEATURES = home_model.get_booster().feature_names or []

class MatchData(BaseModel):
    home_team: str
    away_team: str

def build_feature_row(home_team: str, away_team: str) -> pd.DataFrame:
    """
    Your model was trained on pd.get_dummies(['home_team','away_team']).
    We recreate a single-row dummy vector by:
      - creating a zero vector for all expected features
      - setting 1 on 'home_team_<name>' and 'away_team_<name>' if those columns exist
    """
    if not FEATURES:
        raise HTTPException(500, "Model feature names are empty. Re-train models.")

    row = pd.Series(0.0, index=FEATURES, dtype=float)

    ht_col = f"home_team_{home_team}"
    at_col = f"away_team_{away_team}"

    if ht_col in row.index:
        row[ht_col] = 1.0
    # If team not seen at training, column won’t exist → stays all zeros (baseline)
    if at_col in row.index:
        row[at_col] = 1.0

    return pd.DataFrame([row])

@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(match: MatchData):
    X = build_feature_row(match.home_team, match.away_team)

    # Align columns to booster features (safety)
    booster_features = home_model.get_booster().feature_names or list(X.columns)
    X = X.reindex(columns=booster_features, fill_value=0)

    h = float(home_model.predict(X)[0])
    a = float(away_model.predict(X)[0])

    return {"pred_home_goals": round(h, 3), "pred_away_goals": round(a, 3)}
