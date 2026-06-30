"""Project pipelines."""

from kedro.pipeline import Pipeline

from mlops_project.pipelines import (
    data_drift,
    data_unit_tests,
    data_cleaning,
    feature_selection,
    feature_store,
    ingestion,
    model_predict,
    model_selection,
    model_train,
    preprocessing_batch,
    preprocessing_train,
    split_data,
    split_train,
)

BASE_PIPELINE_FACTORIES = {
    "ingestion": ingestion.create_pipeline,
    "data_unit_tests": data_unit_tests.create_pipeline,
    "data_cleaning": data_cleaning.create_pipeline,
    "split_data": split_data.create_pipeline,
    "split_train": split_train.create_pipeline,
    "preprocessing_train": preprocessing_train.create_pipeline,
    "preprocessing_batch": preprocessing_batch.create_pipeline,
    "feature_selection": feature_selection.create_pipeline,
    "model_selection": model_selection.create_pipeline,
    "model_train": model_train.create_pipeline,
    "model_predict": model_predict.create_pipeline,
    "feature_store": feature_store.create_pipeline,
    "data_drift": data_drift.create_pipeline,
}

DEFAULT_STEPS = (
    "ingestion",
    "data_unit_tests",
    "data_cleaning",
    "feature_store",       
    "split_data",
    "split_train",
    "preprocessing_train",
    "feature_selection",
    "model_selection",
    "model_train",
    "preprocessing_batch",
    "model_predict",
    "data_drift",
)


def _combine(pipelines: dict[str, Pipeline], names: tuple[str, ...]) -> Pipeline:
    return sum((pipelines[name] for name in names), Pipeline([]))


def register_pipelines() -> dict[str, Pipeline]:
    """Register the Green Taxi project pipelines."""
    base_pipelines = {
        name: create_pipeline()
        for name, create_pipeline in BASE_PIPELINE_FACTORIES.items()
    }
    pipelines = base_pipelines.copy()

    pipelines["data_quality"] = _combine(
        base_pipelines,
        ("ingestion", "data_unit_tests"),
    )
    pipelines["production_full_train_process"] = _combine(
        base_pipelines,
        (
            "ingestion",
            "data_cleaning",
            "feature_store",
            "split_data",
            "split_train",
            "preprocessing_train",
            "feature_selection",
            "model_selection",
            "model_train",
        ),
    )
    pipelines["production_full_prediction_process"] = _combine(
        base_pipelines,
        ("preprocessing_batch", "model_predict"),
    )
    pipelines["drift_monitoring"] = _combine(
        base_pipelines,
        (
            "ingestion",
            "data_cleaning",
            "feature_store",
            "split_data",
            "split_train",
            "preprocessing_train",
            "preprocessing_batch",
            "data_drift",
        ),
    )
    pipelines["__default__"] = _combine(base_pipelines, DEFAULT_STEPS)
    return pipelines
