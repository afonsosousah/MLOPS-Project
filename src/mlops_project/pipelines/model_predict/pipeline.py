from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_predict,
                inputs=[
                    "production_model",
                    "production_columns",
                    "val_data",
                    "parameters",
                ],
                outputs="predictions",
                name="model_predict_node",
            ),
        ]
    )
