import os
import pandas as pd
from scipy.stats import ks_2samp
import mlflow
from datetime import datetime
import shutil  # pour remplacer proprement les fichiers

def main():
    print("📊 Début du suivi du Data Drift...")

    processed_path = "data/processed"
    reports_path = "reports"
    os.makedirs(reports_path, exist_ok=True)

    current_path = os.path.join(processed_path, "clean_matches.csv")
    reference_path = os.path.join(processed_path, "reference_data.csv")
    csv_report_path = os.path.join(reports_path, "simple_data_drift_report.csv")
    html_report_path = os.path.join(reports_path, "simple_data_drift_report.html")

    # Charger les données actuelles
    current = pd.read_csv(current_path)

    # Initialiser la référence si elle n’existe pas encore
    if not os.path.exists(reference_path):
        current.to_csv(reference_path, index=False)
        print("🆕 Première exécution — référence initialisée.")
        print("ℹ️ Relance `dvc repro` après collecte de nouvelles données pour mesurer le drift.")
        return

    reference = pd.read_csv(reference_path)

    # --- Comparaison des distributions numériques ---
    common_cols = list(
        set(current.select_dtypes(include="number").columns)
        & set(reference.select_dtypes(include="number").columns)
    )

    if not common_cols:
        print("⚠️ Aucune colonne numérique commune trouvée entre les deux jeux de données.")
        return

    results = []
    for col in common_cols:
        ref_col = reference[col].dropna()
        cur_col = current[col].dropna()
        if len(ref_col) > 0 and len(cur_col) > 0:
            stat, p_value = ks_2samp(ref_col, cur_col)
            drift_detected = p_value < 0.05
            results.append({
                "feature": col,
                "ks_statistic": round(stat, 4),
                "p_value": round(p_value, 4),
                "drift_detected": drift_detected
            })

    drift_df = pd.DataFrame(results)
    drift_df.to_csv(csv_report_path, index=False)

    drift_count = drift_df["drift_detected"].sum()
    drift_rate = drift_count / len(drift_df) if len(drift_df) > 0 else 0

    print(f"✅ Rapport CSV enregistré : {csv_report_path}")
    print(f"📉 {drift_count}/{len(drift_df)} features ont un drift détecté ({drift_rate:.1%})")

    # --- Génération HTML simple ---
    html_content = f"""
    <html>
    <head><title>Data Drift Report</title></head>
    <body>
        <h1>⚙️ Football Prediction - Data Drift Report</h1>
        <p><b>Date :</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><b>Total features :</b> {len(drift_df)}</p>
        <p><b>Drift détecté sur :</b> {drift_count} features ({drift_rate:.1%})</p>
        <hr>
        {drift_df.to_html(index=False)}
    </body>
    </html>
    """
    with open(html_report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Rapport HTML enregistré : {html_report_path}")

    # --- Journalisation MLflow ---
    mlflow.set_experiment("football_prediction_mlops")
    with mlflow.start_run(run_name="data_drift_monitoring"):
        mlflow.log_metric("drifted_features", int(drift_count))
        mlflow.log_metric("drift_rate", drift_rate)
        mlflow.log_artifact(csv_report_path)
        mlflow.log_artifact(html_report_path)

    print("📦 Résultats du drift enregistrés dans MLflow")

    # --- Auto-refresh de la référence ---
    THRESHOLD = 0.3  # 30% de colonnes en drift
    if drift_rate > THRESHOLD:
        backup_path = reference_path.replace(".csv", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        shutil.copy(reference_path, backup_path)
        current.to_csv(reference_path, index=False)
        print(f"🔁 Drift > {THRESHOLD:.0%} détecté — mise à jour automatique de la référence.")
        print(f"📂 Ancienne référence sauvegardée : {backup_path}")
    else:
        print("✅ Aucun drift majeur détecté — référence conservée.")

    print("🎯 Surveillance du Data Drift terminée avec succès !")

if __name__ == "__main__":
    main()
