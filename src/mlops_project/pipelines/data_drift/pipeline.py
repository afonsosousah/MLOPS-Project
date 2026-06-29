from kedro.pipeline import Pipeline, node, pipeline

from .nodes import detect_drift


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=detect_drift,
                inputs=[
                    "X_train_preprocessed",
                    "X_batch_preprocessed",
                    "params:data_drift",
                ],
                outputs="drift_report",
                name="detect_drift_node",
            ),
        ]
    )
