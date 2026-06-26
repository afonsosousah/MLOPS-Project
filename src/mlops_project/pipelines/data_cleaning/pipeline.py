from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_green_taxi_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_green_taxi_data,
                inputs=["ingested_data", "params:data_cleaning"],
                outputs="cleaned_data",
                name="clean_green_taxi_data_node",
            ),
        ]
    )