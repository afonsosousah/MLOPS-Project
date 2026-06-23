from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_batch


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_batch,
                inputs=["analysis_data", "encoder_transform"],
                outputs="preprocessed_batch_data",
                name="preprocessing_batch_node",
            ),
        ]
    )
