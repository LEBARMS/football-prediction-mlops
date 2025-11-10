#   Football Match Prediction — End-to-End MLOps Project

##  Objectif du projet

Ce projet met en œuvre un pipeline MLOps complet pour la prédiction des scores de matchs de football
à partir de données collectées automatiquement sur Football-Data.co.uk

L’objectif est d’illustrer une approche reproductible et automatisée intégrant :
```

-la collecte et le prétraitement des données,

-l’entraînement et le suivi des modèles,

-la génération des prédictions,

-la surveillance du Data Drift avec journalisation dans MLflow.
.

```
##  Architecture du projet
```
football-prediction-mlops/
│
├── data/
│   ├── raw/              # Données brutes collectées depuis Football-Data
│   ├── processed/        # Données nettoyées et enrichies
│   └── predictions/      # Prédictions finales du modèle
│
├── src/
│   ├── fetch_data_universal.py   # Collecte multi-ligues et multi-saisons (Football-Data)
│   ├── preprocess.py             # Nettoyage et fusion des données
│   ├── train.py                  # Entraînement du modèle XGBoost + MLflow
│   ├── predict.py                # Génération et évaluation des prédictions
│   └── monitor_drift.py          # Détection automatique du Data Drift
│
├── models/                       # Modèles XGBoost sauvegardés
│
├── reports/                      # Rapports de drift CSV + HTML
│
├── dvc.yaml                       # Définition du pipeline DVC
├── dvc.lock                       # Suivi des dépendances et outputs DVC
├── requirements.txt               # Dépendances du projet
└── .gitignore                     # Fichiers à exclure du suivi Git
```
---

## ⚙️ Installation et exécution

### 1 Cloner le dépôt
```bash
git clone https://github.com/essokri/football-prediction-mlops.git
cd football-prediction-mlops

```
### 2 Créer et activer un environnement virtuel
 ```
 python -m venv .venv
.venv\Scripts\activate       # (Windows)
# ou
source .venv/bin/activate    # (Linux / Mac)
 
 ```
### 3 Installer les dépendances
 ```
 pip install -r requirements.txt

```
### 4 Lancer le pipeline complet
```
 dvc repro

 Cette commande exécute automatiquement toutes les étapes :
 fetch_data_universal → preprocess → train → predict → monitor_drift

```
### 5 Visualiser les résultats
```
 # Sorties principales du pipeline :
- Données nettoyées : data/processed/clean_matches.csv
- Prédictions finales : data/predictions/predicted_matches.csv
- Rapports de Data Drift :
    • CSV  → reports/simple_data_drift_report.csv
    • HTML → reports/simple_data_drift_report.html
- Modèles entraînés : models/home_model.json et models/away_model.json

#  Suivi des expériences et métriques :
mlflow ui
# Puis ouvrir dans le navigateur :
http://localhost:5000 

```

## Étapes actuelles implémentées
```
- Collecte automatique des données multi-ligues et multi-saisons depuis Football-Data.co.uk
- Prétraitement, nettoyage et fusion des matchs avec les statistiques d’équipes
- Entraînement des modèles XGBoost pour la prédiction des buts à domicile et à l’extérieur
- Suivi des expériences et métriques avec MLflow
- Génération et évaluation des prédictions finales
- Détection et rapport du Data Drift
- Gestion et orchestration complète du pipeline avec DVC

```
