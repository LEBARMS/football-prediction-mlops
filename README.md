#   Football Match Prediction — End-to-End MLOps Project

##  Objectif du projet

Ce projet vise à construire un pipeline **MLOps complet** pour prédire le résultat d’un match de football (victoire, nul, défaite) à partir de données collectées automatiquement depuis **SoccerData** (FBref, Understat, etc.).
.


##  Architecture du projet
```
football-prediction-mlops/
│
├── data/
│ ├── raw/ # Données brutes collectées (non suivies par Git)
│ ├── processed/ # Données nettoyées
│ └── predictions/ # Résultats et prédictions
│
├── src/
│ ├── fetch_data.py # Collecte automatique des données via SoccerData
│ ├── preprocess.py # Prétraitement et nettoyage des données
│ ├── train.py # Entraînement du modèle + suivi MLflow
│ └── predict.py # Génération des prédictions finales
│
├── models/ # Modèles sauvegardés (non suivis par Git)
├── dvc.yaml # Pipeline DVC
├── dvc.lock # Suivi des versions des étapes DVC
├── requirements.txt # Dépendances du projet
└── .gitignore
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
 fetch_data → preprocess → train → predict 

```
### 5 Visualiser les résultats
```
 Les prédictions finales se trouvent dans : data/predictions/predicted_matches.csv
 Les métriques et modèles sont suivis via MLflow : mlflow ui
 Ouvrir http://localhost:5000 

```
### 6 Lancer l'API REST
```
uvicorn app.main:app --reload
Ouvrir :  http://127.0.0.1:8000
Cliquer sur post/predict
Cliquer sur try it out
Changer le nom du Home Team et du Away Team
Execute
Résultat quelques cases vers le bas
```
### 7 Docker build & run
```
docker build -t football-api:0.1.0 .
docker run --rm -p 8000:8000 football-api:0.1.0
```

## Étapes actuelles implémentées
```
- Collecte automatique des données (SoccerData)
- Prétraitement et fusion des fichiers CSV
- Entraînement du modèle XGBoost avec MLflow
- Génération et évaluation des prédictions
- Suivi complet du pipeline avec DVC
```
