from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    engineer_features,
    select_features,
    upload_to_hopsworks
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=engineer_features,
                inputs="cleaned_data",
                outputs="engineered_data",
                name="engineer_features_node",
            ),
            node(
                func=select_features,
                inputs="engineered_data",
                outputs="feature_store_data",
                name="select_features_node",
            ),
            node(
                func=upload_to_hopsworks,
                inputs=["feature_store_data", "parameters"],
                outputs= None,
                name="upload_to_hopsworks_node"
            )
        ]
    )