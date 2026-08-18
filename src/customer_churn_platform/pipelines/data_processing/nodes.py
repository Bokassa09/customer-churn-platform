"""
Pipeline data_processing
Preprocessing des données de churn bancaire
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_clean_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Nettoyer les données brutes"""
    # Supprimer les colonnes inutiles
    df_clean = raw_data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    return df_clean


def split_data(clean_data: pd.DataFrame) -> tuple:
    """Séparer features et target, puis split train/test"""
    X = clean_data.drop('Exited', axis=1)
    y = clean_data['Exited']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test