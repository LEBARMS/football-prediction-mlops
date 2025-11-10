import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.xgboost
import os

def main():
    print("🚀 Démarrage de l’entraînement des modèles XGBoost...")

    # 1️⃣ Charger les données prétraitées
    data_path = "data/processed/clean_matches.csv"
    data = pd.read_csv(data_path)
    print(f"✅ Données chargées : {data.shape[0]} matchs, {data.shape[1]} colonnes")

    # 2️⃣ Sélection des features numériques pertinentes
    features = [
        "home_matches_played", "home_goals_for", "home_goals_against", "home_goals_diff",
        "away_matches_played", "away_goals_for", "away_goals_against", "away_goals_diff"
    ]

    for f in features:
        if f not in data.columns:
            raise ValueError(f"⚠️ La colonne '{f}' est manquante dans les données.")

    X = data[features]
    y_home = data["home_goals"]
    y_away = data["away_goals"]

    # 3️⃣ Division train/test
    X_train, X_test, y_home_train, y_home_test = train_test_split(
        X, y_home, test_size=0.2, random_state=42
    )
    _, _, y_away_train, y_away_test = train_test_split(
        X, y_away, test_size=0.2, random_state=42
    )

    # 4️⃣ Configuration MLflow
    mlflow.set_experiment("football_prediction_mlops")
    with mlflow.start_run(run_name="xgboost_multi_leagues"):
        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_param("features", features)
        mlflow.log_param("test_size", 0.2)

        # 5️⃣ Entraînement des modèles
        model_home = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        model_away = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )

        model_home.fit(X_train, y_home_train)
        model_away.fit(X_train, y_away_train)

        # 6️⃣ Évaluation
        y_home_pred = model_home.predict(X_test)
        y_away_pred = model_away.predict(X_test)

        metrics = {
            "mse_home": mean_squared_error(y_home_test, y_home_pred),
            "mae_home": mean_absolute_error(y_home_test, y_home_pred),
            "r2_home": r2_score(y_home_test, y_home_pred),
            "mse_away": mean_squared_error(y_away_test, y_away_pred),
            "mae_away": mean_absolute_error(y_away_test, y_away_pred),
            "r2_away": r2_score(y_away_test, y_away_pred)
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        print("📊 Résultats du modèle :")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # 7️⃣ Sauvegarde des modèles
        os.makedirs("models", exist_ok=True)
        home_model_path = "app/model/home_model.json"
        away_model_path = "app/model/away_model.json"
        model_home.save_model(home_model_path)
        model_away.save_model(away_model_path)

        mlflow.log_artifact(home_model_path)
        mlflow.log_artifact(away_model_path)

        print("✅ Modèles sauvegardés et enregistrés dans MLflow.")

if __name__ == "__main__":
    main()
