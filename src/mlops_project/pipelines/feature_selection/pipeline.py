from kedro.pipeline import Pipeline, node, pipeline

from .nodes import select_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=select_features,
                inputs=[
                    "X_train_preprocessed",
                    "y_train_data",
                    "params:feature_selection",
                ],
                outputs="best_columns",
                name="select_features_node",
            )
        ]
    )
