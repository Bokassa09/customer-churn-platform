"""
Pipeline explainability
SHAP values pour expliquer les prédictions
"""
import pandas as pd
import numpy as np
import shap
from catboost import CatBoostClassifier


def compute_shap_values(model: CatBoostClassifier,
                        X_test: pd.DataFrame) -> pd.DataFrame:
    """Calculer les valeurs SHAP sur le test set"""

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Créer un DataFrame avec les valeurs SHAP
    shap_df = pd.DataFrame(
        shap_values,
        columns=X_test.columns
    )

    print(f"SHAP values calculées sur {len(X_test)} exemples")
    print(f"Feature la plus importante : {shap_df.abs().mean().idxmax()}")

    return shap_df