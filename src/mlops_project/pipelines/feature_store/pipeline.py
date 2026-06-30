from kedro.pipeline import Pipeline, node, pipeline

from .nodes import upload_features_to_store


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=upload_features_to_store,
                inputs=["data_clean", "params:feature_store"],
                outputs="feature_store_upload_report",
                name="upload_features_to_store_node",
            )
        ]
    )
