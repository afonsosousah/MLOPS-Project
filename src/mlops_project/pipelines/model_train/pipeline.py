from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_train,
                inputs=["train_data", "val_data", "best_model_params", "parameters"],
                outputs=[
                    "production_model",
                    "production_columns",
                    "production_model_metrics",
                    "validation_predictions",
                ],
                name="model_train_node",
            ),
        ]
    )