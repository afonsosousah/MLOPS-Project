"""Project pipelines."""

from kedro.pipeline import Pipeline

from mlops_project.pipelines import (
    ingestion,
    data_unit_tests,
    data_cleaning,
    feature_engineering,
    feature_selection,
    split_data,
    model_train,
)

ACTIVE_PIPELINE_FACTORIES = {
    "ingestion": ingestion.create_pipeline,
    "data_unit_tests": data_unit_tests.create_pipeline,
    "data_cleaning": data_cleaning.create_pipeline,
    "feature_engineering": feature_engineering.create_pipeline,
    "feature_selection": feature_selection.create_pipeline,
    "split_data": split_data.create_pipeline,
    "model_train": model_train.create_pipeline,
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
