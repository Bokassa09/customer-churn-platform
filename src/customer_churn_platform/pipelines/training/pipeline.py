"""
Pipeline training
"""
from kedro.pipeline import Pipeline, node
from .nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([

        node(
            func=train_model,
            inputs=["X_train", "X_test", "y_train", "y_test"],
            outputs=["model", "metrics"],
            name="train_model_node"
        ),

    ])