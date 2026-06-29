from kedro.pipeline import Pipeline, node, pipeline

from .nodes import ingestion


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=ingestion,
                inputs=[
                    "green_taxi_raw",
                    "zone_lookup",
                    "params:modeling_years",
                    "params:drift_years",
                ],
                outputs=["ingested_data", "drift_ingested_data"],
                name="ingestion_node",
            ),
        ]
    )
