from kedro.pipeline import Pipeline, node, pipeline

from .nodes import detect_drift


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=detect_drift,
                inputs=[
                    "X_train_data",
                    "X_train_preprocessed",
                    "drift_ingested_data",
                    "preprocessing_transformer",
                    "production_columns",
                    "params:data_cleaning",
                    "params:target_column",
                    "params:data_drift",
                ],
                outputs=["drift_report", "drift_summary", "X_drift_preprocessed"],
                name="detect_drift_node",
            ),
        ]
    )
