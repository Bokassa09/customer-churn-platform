# 🏦 Customer Churn Intelligence Platform

> Pipeline MLOps complet de détection et d'explication du churn bancaire

[![CI/CD](https://github.com/Bokassa09/customer-churn-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Bokassa09/customer-churn-platform/actions)

---

## Problématique métier

Une banque constate que des clients ferment leur compte chaque mois.
Acquérir un nouveau client coûte **5 à 7 fois plus cher** que d'en retenir un.
Ce projet identifie à l'avance les clients à risque de départ (churn)
et explique les raisons pour cibler les actions de rétention.

---

## Demo en ligne

| Service | Lien |
|---------|------|
| 🖥️ Dashboard | [customer-churn-platform-1-zeg0.onrender.com](https://customer-churn-platform-1-zeg0.onrender.com) |
| ⚡ API FastAPI | [customer-churn-platform-f0g1.onrender.com/docs](https://customer-churn-platform-f0g1.onrender.com/docs) |

> ⚠️ Le free tier Render peut mettre 50 secondes à démarrer après inactivité.

---

## Résultats du modèle

| Métrique | Score |
|----------|-------|
| ROC-AUC | **87.5%** |
| Recall | **69.0%** |
| Precision | **54.7%** |
| F1 Score | **61.0%** |

Modèle sélectionné selon le **Recall** : métrique prioritaire pour ne pas rater de vrais churners.

---

## Stack technique

```
ML          → CatBoost + Optuna (optimisation automatique)
Explicabilité → SHAP (Top 3 facteurs par client)
LLM         → Groq Compound Mini (recommandations conseiller)
MLOps       → MLflow (Tracking + Model Registry)
Pipeline    → Kedro (structure modulaire reproductible)
API         → FastAPI
Dashboard   → Dash (Plotly)
CI/CD       → GitHub Actions (pytest)
Déploiement → Docker + Render
```

---

## Architecture

```
Données CSV
    ↓ Kedro Pipeline
Preprocessing → CatBoost + Optuna → SHAP → MLflow Registry
    ↓
FastAPI (/predict + /explain + /analyze)
    ↓ Groq LLM
Dashboard Dash
    ↓
Déployé sur Render (2 services Docker)
```

---

## 📁 Structure du projet

```
customer-churn-platform/
├── main.py                    # API FastAPI
├── app.py                     # Dashboard Dash
├── model.pkl                  # Modèle CatBoost
├── Dockerfile                 # Image API
├── Dockerfile.dash            # Image Dashboard
├── requirements-api.txt       # Dépendances API
├── requirements-dash.txt      # Dépendances Dashboard
├── .github/workflows/ci.yml   # Pipeline CI/CD
├── notebooks/
│   ├── 01_eda.ipynb           # Exploration des données
│   ├── 02_modeling.ipynb      # CatBoost + Optuna
│   ├── 03_shap.ipynb          # Explicabilité SHAP
│   ├── 04_mlflow.ipynb        # Tracking MLflow
│   └── 05_registry.py         # Model Registry
├── src/pipelines/
│   ├── data_processing/       # Pipeline Kedro
│   ├── training/
│   └── explainability/
└── tests/
    └── test_api.py            # Tests automatiques
```

---

## Lancer en local

```bash
# Cloner le repo
git clone https://github.com/Bokassa09/customer-churn-platform.git
cd customer-churn-platform

# Créer l'environnement
python3 -m venv env
source env/bin/activate
pip install -r requirements-api.txt

# Variables d'environnement
echo "GROQ_API_KEY=ta-clé-groq" > .env

# Lancer l'API
uvicorn main:app --reload

# Lancer le dashboard (autre terminal)
pip install dash requests
python app.py
```

---

## 👤 Auteur

**Omer Bokassa Boueke** : ML Engineer & MLOps
