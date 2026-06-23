from kedro.pipeline import Pipeline, node, pipeline

from .nodes import ingestion


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=ingestion,
                inputs=["green_taxi_raw", "zone_lookup", "parameters"],
                outputs="ingested_data",
                name="ingestion_node",
            ),
        ]
    )
