from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=[
                    "ingested_data",
                    "params:train_test_split_date",
                    "params:preprocessing",
                ],
                outputs=["train_data", "test_data"],
                name="split_data_node",
            ),
        ]
    )
