from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_predict,
                inputs=[
                    "X_batch_preprocessed",
                    "production_model",
                    "production_columns",
                ],
                outputs=["predictions", "predict_describe"],
                name="model_predict_node",
            ),
        ]
    )
