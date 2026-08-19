import mlflow.catboost
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import shap
import mlflow

import joblib

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=api_key) if api_key else None


feature_names = ['CreditScore', 'Geography', 'Gender', 'Age', 
                 'Tenure', 'Balance', 'NumOfProducts', 
                 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Charger le modèle depuis MLflow Registry
#mlflow.set_tracking_uri("sqlite:////app/notebooks/mlflow.db")
#model = mlflow.catboost.load_model(
#    "models:/customer-churn-classifier@production")

model = joblib.load("model.pkl")

# Explainer SHAP
explainer = shap.TreeExplainer(model)

app = FastAPI(
    title="Customer Churn Intelligence Platform",
    description="Détection et explication du churn bancaire",
    version="1.0"
)

# Format des données client
class ClientData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# Route accueil
@app.get("/")
def accueil():
    return {
        "bienvenue": "Bienvenue ! Ce modèle permet de détecter et d'expliquer le churn bancaire.",
        "description": "Entrez les 10 features du client.",
        "routes": {
            "/predict": "Prédire le churn d'un client",
            "/stats": "Statistiques globales des prédictions",
            "/explication": "Expliquer la décision du modèle",
            "/health": "Vérifier que l'API fonctionne",
        }
    }

# Route health
@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "API en ligne et opérationnelle",
        "version": "1.0"
    }


# Route predict 
@app.post("/predict")
def predict(data: ClientData):

    X = pd.DataFrame([{
    'CreditScore': data.CreditScore,
    'Geography': data.Geography,
    'Gender': data.Gender,
    'Age': data.Age,
    'Tenure': data.Tenure,
    'Balance': data.Balance,
    'NumOfProducts': data.NumOfProducts,
    'HasCrCard': data.HasCrCard,
    'IsActiveMember': data.IsActiveMember,
    'EstimatedSalary': data.EstimatedSalary
}])

    prediction = model.predict(X)[0]
    probabilite = model.predict_proba(X)[0].max()
    pourcentage = round(float(probabilite) * 100, 1)

    if prediction == 1:
        message = f"Churn détecté, avec une confiance de {pourcentage} %"
    else:
        message = f"Ce client est fidèle, avec une confiance de {pourcentage} %"

    return {
        "prediction": int(prediction),
        "probabilite": pourcentage,
        "message": message
    }

# Route explain
@app.post("/explain")
def explain(data: ClientData):
    X = pd.DataFrame([{
        'CreditScore': data.CreditScore,
        'Geography': data.Geography,
        'Gender': data.Gender,
        'Age': data.Age,
        'Tenure': data.Tenure,
        'Balance': data.Balance,
        'NumOfProducts': data.NumOfProducts,
        'HasCrCard': data.HasCrCard,
        'IsActiveMember': data.IsActiveMember,
        'EstimatedSalary': data.EstimatedSalary
    }])

    shap_vals = explainer.shap_values(X)

    # Créer dictionnaire feature 
    shap_dict = {
        feat: round(float(val), 4)
        for feat, val in zip(feature_names, shap_vals[0])
    }

    # Top 3 
    top3 = dict(sorted(
        shap_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3])

    # Messages métier
    messages = {
        'Age':            "Âge élevé : client proche de la retraite",
        'IsActiveMember': "Client inactif : risque de décrochage",
        'Balance':        "Solde élevé : courtisé par la concurrence",
        'Geography':      "Région à fort taux de churn",
        'NumOfProducts':  "Nombre de produits atypique",
        'Gender':         "Profil démographique à risque",
        'CreditScore':    "Score de crédit influent",
        'Tenure':         "Ancienneté faible",
        'HasCrCard':      "Possession de carte de crédit",
        'EstimatedSalary':"Salaire influent"
    }

    return {
        "base_value": round(float(explainer.expected_value), 4),
        "top3_facteurs": {
            feat: {
                "shap": val,
                "sens": "↑ churn" if val > 0 else "↓ churn",
                "explication": messages.get(feat, feat)
            }
            for feat, val in top3.items()
        }
    }


def analyze(data: ClientData):
    if groq_client is None:
        return {"analyse_llm": "Service LLM non disponible"}

    # Prédiction
    X = pd.DataFrame([{
        'CreditScore': data.CreditScore,
        'Geography': data.Geography,
        'Gender': data.Gender,
        'Age': data.Age,
        'Tenure': data.Tenure,
        'Balance': data.Balance,
        'NumOfProducts': data.NumOfProducts,
        'HasCrCard': data.HasCrCard,
        'IsActiveMember': data.IsActiveMember,
        'EstimatedSalary': data.EstimatedSalary
    }])

    prediction = model.predict(X)[0]
    probabilite = model.predict_proba(X)[0].max()
    pourcentage = round(float(probabilite) * 100, 1)

    # SHAP
    shap_vals = explainer.shap_values(X)
    shap_dict = {
        feat: round(float(val), 4)
        for feat, val in zip(feature_names, shap_vals[0])
    }
    top3 = dict(sorted(
        shap_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3])

    # Prompt pour Groq
    statut = "va quitter la banque" if prediction == 1 else "va rester"
    top3_texte = "\n".join([
        f"- {feat} : {val:+.2f}" 
        for feat, val in top3.items()
    ])

    prompt = f"""Tu es un expert en risque bancaire. 
Analyse ce profil client de manière concise et professionnelle.

Client : {data.Age} ans, {data.Gender}, {data.Geography}
Solde : {data.Balance}€ | Produits : {data.NumOfProducts}
Actif : {'Oui' if data.IsActiveMember else 'Non'}

Prédiction : {pourcentage}% de risque de churn : ce client {statut}.

Facteurs principaux (valeurs SHAP) :
{top3_texte}

Génère en 3-4 phrases :
1. Une analyse du profil
2. Les raisons principales du risque
3. Une recommandation concrète pour le conseiller bancaire

Réponds en français, de manière professionnelle et actionnable."""

    # Appel Groq
    response = groq_client.chat.completions.create(
        model="groq/compound-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7
    )

    analyse = response.choices[0].message.content

    return {
        "prediction": int(prediction),
        "probabilite": pourcentage,
        "analyse_llm": analyse
    }