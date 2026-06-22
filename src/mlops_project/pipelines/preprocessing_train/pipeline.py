from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_train,
                inputs="ref_data",
                outputs=["preprocessed_training_data", "encoder_transform"],
                name="preprocessing_train_node",
            ),
        ]
    )
