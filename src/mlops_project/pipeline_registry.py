"""Project pipelines."""

from kedro.pipeline import Pipeline

from mlops_project.pipelines import (
    data_cleaning,
    data_unit_tests,
    ingestion,
    preprocessing_batch,
    preprocessing_train,
    split_data,
)

ACTIVE_PIPELINE_FACTORIES = {
    "ingestion": ingestion.create_pipeline,
    "data_unit_tests": data_unit_tests.create_pipeline,
    "data_cleaning": data_cleaning.create_pipeline,
    "split_data": split_data.create_pipeline,
    "preprocessing_train": preprocessing_train.create_pipeline,
    "preprocessing_batch": preprocessing_batch.create_pipeline,
}


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    active_pipelines = {
        name: create_pipeline()
        for name, create_pipeline in ACTIVE_PIPELINE_FACTORIES.items()
    }
    pipelines = active_pipelines.copy()

    # data_unit_tests runs ingestion first so the validation node has its input
    if {"ingestion", "data_unit_tests"}.issubset(active_pipelines):
        pipelines["data_unit_tests"] = (
            active_pipelines["ingestion"]
            + active_pipelines["data_unit_tests"]
        )

    # Full data preparation chain: ingest → split → preprocess (produces model inputs)
    if {"ingestion", "split_data", "preprocessing_train"}.issubset(active_pipelines):
        pipelines["data_prep"] = (
            active_pipelines["ingestion"]
            + active_pipelines["split_data"]
            + active_pipelines["preprocessing_train"]
        )

    pipelines["__default__"] = sum(active_pipelines.values(), Pipeline([]))
    return pipelines
