from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_selection


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_selection,
                inputs=[
                    "X_train_preprocessed",
                    "X_val_preprocessed",
                    "y_train_data",
                    "y_val_data",
                    "best_columns",
                    "params:model_selection",
                ],
                outputs="selected_model_metadata",
                name="model_selection_node",
            ),
        ]
    )
