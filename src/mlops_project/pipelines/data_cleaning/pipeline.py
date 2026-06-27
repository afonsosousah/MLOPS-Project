from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_data,
                inputs=["modeling_ingested_data", "params:data_cleaning"],
                outputs="modeling_cleaned_data",
                name="clean_modeling_data_node",
            ),
        ]
    )