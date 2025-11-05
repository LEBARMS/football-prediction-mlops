import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def main():
    print("🔮 Début des prédictions...")

    processed_path = "data/processed"
    model_path = "app/model"
    pred_path = "data/predictions"
    os.makedirs(pred_path, exist_ok=True)

    # 1️⃣ Charger les données et les modèles
    data = pd.read_csv(os.path.join(processed_path, "clean_matches.csv"))

    home_model = xgb.XGBRegressor()
    away_model = xgb.XGBRegressor()
    home_model.load_model(os.path.join(model_path, "home_model.json"))
    away_model.load_model(os.path.join(model_path, "away_model.json"))

    print(f"✅ Données chargées : {len(data)} matchs")

    # 2️⃣ Préparer les données comme à l'entraînement
    X = pd.get_dummies(data, columns=["home_team", "away_team"], drop_first=False)
    expected_features = home_model.get_booster().feature_names
    X = X.reindex(columns=expected_features, fill_value=0)

    # 3️⃣ Faire des prédictions
    data["pred_home_goals"] = home_model.predict(X)
    data["pred_away_goals"] = away_model.predict(X)

    # 4️⃣ Calculer les métriques
    mse_home = mean_squared_error(data["home_goals"], data["pred_home_goals"])
    mae_home = mean_absolute_error(data["home_goals"], data["pred_home_goals"])
    r2_home = r2_score(data["home_goals"], data["pred_home_goals"])

    mse_away = mean_squared_error(data["away_goals"], data["pred_away_goals"])
    mae_away = mean_absolute_error(data["away_goals"], data["pred_away_goals"])
    r2_away = r2_score(data["away_goals"], data["pred_away_goals"])

    print(f"📊 MSE Home: {mse_home:.3f}, MAE Home: {mae_home:.3f}, R² Home: {r2_home:.3f}")
    print(f"📊 MSE Away: {mse_away:.3f}, MAE Away: {mae_away:.3f}, R² Away: {r2_away:.3f}")

    # 5️⃣ Sauvegarder les prédictions
    output_file = os.path.join(pred_path, "predicted_matches.csv")
    data.to_csv(output_file, index=False)
    print(f"✅ Prédictions enregistrées dans {output_file}")

    # 6️⃣ Enregistrer tout dans MLflow
    mlflow.set_experiment("football_prediction")
    with mlflow.start_run(run_name="prediction_run"):
        mlflow.log_metric("mse_home", mse_home)
        mlflow.log_metric("mae_home", mae_home)
        mlflow.log_metric("r2_home", r2_home)
        mlflow.log_metric("mse_away", mse_away)
        mlflow.log_metric("mae_away", mae_away)
        mlflow.log_metric("r2_away", r2_away)
        mlflow.log_artifact(output_file)  # pour voir le CSV dans MLflow
        print("📦 Résultats et prédictions enregistrés dans MLflow")

    print("🎯 Prédiction terminée avec succès !")

if __name__ == "__main__":
    main()
