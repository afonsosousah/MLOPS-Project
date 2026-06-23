from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_train,
                inputs=["ref_data", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test", "encoder_transform"],
                name="preprocessing_train_node",
            ),
        ]
    )
