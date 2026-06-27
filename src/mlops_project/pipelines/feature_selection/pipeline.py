from kedro.pipeline import Pipeline, node, pipeline

from .nodes import select_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=select_features,
                inputs="modeling_engineered_data",
                outputs="modeling_selected_features_data",
                name="select_features_node",
            )
        ]
    )