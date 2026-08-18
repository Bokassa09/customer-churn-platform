"""
Pipeline data_processing
"""
from kedro.pipeline import Node, Pipeline, node
from .nodes import load_and_clean_data, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([

        node(
            func=load_and_clean_data,
            inputs="raw_data",
            outputs="clean_data",
            name="load_and_clean_data_node"
        ),

        node(
            func=split_data,
            inputs="clean_data",
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node"
        ),

    ])