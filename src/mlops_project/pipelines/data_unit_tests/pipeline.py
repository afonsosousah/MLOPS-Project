from kedro.pipeline import Pipeline, node, pipeline

from .nodes import unit_test


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=unit_test,
                inputs="ingested_data",
                outputs="reporting_tests",
                name="data_unit_tests_node",
            ),
        ]
    )
