"""Project pipelines."""

from kedro.pipeline import Pipeline

from mlops_project.pipelines import (
    data_cleaning,
    data_unit_tests,
    feature_engineering,
    feature_selection,
    ingestion,
    model_train,
    split_data,
    model_predict,
    data_drift,
)

ACTIVE_PIPELINE_FACTORIES = {
    "ingestion": ingestion.create_pipeline,
    "data_unit_tests": data_unit_tests.create_pipeline,
    "data_cleaning": data_cleaning.create_pipeline,
    "feature_engineering": feature_engineering.create_pipeline,
    "feature_selection": feature_selection.create_pipeline,
    "split_data": split_data.create_pipeline,
    "model_train": model_train.create_pipeline,
    "model_predict": model_predict.create_pipeline,
    "data_drift": data_drift.create_pipeline,
}


def register_pipelines() -> dict[str, Pipeline]:
    """Register the Green Taxi project pipelines."""
    active_pipelines = {
        name: create_pipeline()
        for name, create_pipeline in ACTIVE_PIPELINE_FACTORIES.items()
    }
    pipelines = active_pipelines.copy()

    # data_unit_tests runs ingestion first so validation nodes have inputs.
    if {"ingestion", "data_unit_tests"}.issubset(active_pipelines):
        pipelines["data_unit_tests"] = (
            active_pipelines["ingestion"] + active_pipelines["data_unit_tests"]
        )

    data_prep_steps = [
        "ingestion",
        "data_cleaning",
        "feature_engineering",
        "feature_selection",
        "split_data",
    ]
    if set(data_prep_steps).issubset(active_pipelines):
        pipelines["data_prep"] = sum(
            (active_pipelines[name] for name in data_prep_steps),
            Pipeline([]),
        )

    if {"data_prep", "model_train"}.issubset(pipelines):
        pipelines["training"] = pipelines["data_prep"] + active_pipelines["model_train"]

    pipelines["__default__"] = sum(active_pipelines.values(), Pipeline([]))
    return pipelines
