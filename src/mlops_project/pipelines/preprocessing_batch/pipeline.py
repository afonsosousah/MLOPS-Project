from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocessing_batch


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocessing_batch,
                inputs=[
                    "test_data",
                    "preprocessing_transformer",
                    "production_columns",
                ],
                outputs="X_batch_preprocessed",
                name="preprocessing_batch_node",
            ),
        ]
    )
