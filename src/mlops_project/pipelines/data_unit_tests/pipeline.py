from kedro.pipeline import Node, Pipeline

from .nodes import unit_test


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=unit_test,
                inputs="green_taxi_raw",
                outputs=None,
                name="unit_data_test_node"
            )
        ]
    )