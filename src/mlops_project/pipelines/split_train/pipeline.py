from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_train,
                inputs=[
                    "train_data",
                    "params:train_val_split_date",
                    "params:target_column",
                ],
                outputs=[
                    "X_train_data",
                    "X_val_data",
                    "y_train_data",
                    "y_val_data",
                ],
                name="split_train_node",
            ),
        ]
    )
