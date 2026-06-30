from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data, get_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_features,
                inputs=["params:feature_store", "data_clean"],
                outputs="features_from_store",
                name="get_features_node",
            ),
            node(
                func=split_data,
                inputs=["features_from_store", "params:train_test_split_date"],
                outputs=["train_data", "test_data"],
                name="split_data_node",
            ),
        ]
    )