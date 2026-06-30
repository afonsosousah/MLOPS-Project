from kedro.pipeline import Pipeline, node, pipeline

from .nodes import data_cleaning


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=data_cleaning,
                inputs=["ingested_data", "params:data_cleaning"],
                outputs="data_clean",
                name="clean_data_node",
            ),
        ]
    )