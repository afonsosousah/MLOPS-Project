from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    engineer_features,
    upload_to_hopsworks
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=engineer_features,
                inputs="modeling_cleaned_data",
                outputs="modeling_engineered_data",
                name="engineer_features_node",
            ),
            node(
                func=upload_to_hopsworks,
                inputs=["modeling_engineered_data", "parameters"],
                outputs= None,
                name="upload_to_hopsworks_node"
            )
        ]
    )