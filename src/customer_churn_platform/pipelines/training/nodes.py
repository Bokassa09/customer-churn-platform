"""
Pipeline training
Entraînement CatBoost avec MLflow tracking
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.catboost
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)


def train_model(X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.Series, y_test: pd.Series) -> tuple:
    """Entraîner CatBoost et tracker avec MLflow"""

    cat_features = ['Geography', 'Gender']

    params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'class_weights': {0: 1, 1: 4},
        'cat_features': cat_features,
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': 0
    }

    mlflow.set_tracking_uri("sqlite:///notebooks/mlflow.db")
    mlflow.set_experiment("customer-churn-kedro")

    with mlflow.start_run(run_name="CatBoost_Kedro_Pipeline"):

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_proba),
            "recall":    recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred)
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.catboost.log_model(model, "model")

        print(f"ROC-AUC : {metrics['roc_auc']:.4f}")
        print(f"Recall  : {metrics['recall']:.4f}")
        print(f"F1      : {metrics['f1']:.4f}")

    return model, metrics