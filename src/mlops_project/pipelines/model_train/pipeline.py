from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_train,
                inputs=[
                    "X_train_preprocessed",
                    "X_val_preprocessed",
                    "y_train_data",
                    "y_val_data",
                    "best_columns",
                    "selected_model_metadata",
                    "params:model_train",
                    "params:model_selection",
                ],
                outputs=[
                    "production_model",
                    "production_model_metrics",
                    "validation_predictions",
                    "output_plot",
                ],
                name="model_train_node",
            ),
        ]
    )
