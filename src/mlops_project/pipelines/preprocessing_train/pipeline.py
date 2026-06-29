from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocessing_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocessing_train,
                inputs=[
                    "X_train_data",
                    "X_val_data",
                    "params:preprocessing",
                    "params:target_column",
                ],
                outputs=[
                    "X_train_preprocessed",
                    "X_val_preprocessed",
                    "preprocessing_transformer",
                    "production_columns",
                    "reporting_data_train",
                ],
                name="preprocessing_train_node",
            ),
        ]
    )
