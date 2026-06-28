from kedro.pipeline import Pipeline, node, pipeline

from .nodes import detect_drift


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=detect_drift,
                inputs=["modeling_ingested_data", "drift_ingested_data"],
                outputs="drift_report",
                name="detect_drift_node",
            ),
        ]
    )
