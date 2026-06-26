from kedro.pipeline import Pipeline, node, pipeline

from .nodes import unit_test


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=unit_test,
                inputs="modeling_ingested_data",
                outputs="modeling_data_reporting_tests",
                name="modeling_data_unit_tests_node",
            ),
            node(
                func=unit_test,
                inputs="drift_ingested_data",
                outputs="drift_data_reporting_tests",
                name="drift_data_unit_tests_node",
            ),
        ]
    )
