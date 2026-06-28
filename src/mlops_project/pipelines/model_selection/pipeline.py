from kedro.pipeline import Pipeline, node, pipeline

from .nodes import select_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=select_model,
                inputs=["train_data", "val_data", "parameters"],
                outputs="best_model_params",
                name="select_model_node",
            ),
        ]
    )
