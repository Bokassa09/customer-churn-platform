"""
Pipeline explainability
"""
from kedro.pipeline import Pipeline, node
from .nodes import compute_shap_values


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([

        node(
            func=compute_shap_values,
            inputs=["model", "X_test"],
            outputs="shap_df",
            name="compute_shap_node"
        ),

    ])