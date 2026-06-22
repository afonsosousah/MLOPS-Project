from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_by_year


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_by_year,
                inputs="ingested_data",
                outputs=["ref_data", "ana_data"],
                name="split_data_node",
            ),
        ]
    )
