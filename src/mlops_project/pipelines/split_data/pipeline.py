from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["feature_store_data", "params:split_date"],
                outputs=["train_data", "val_data"],
                name="split_data_node",
            ),
        ]
    )