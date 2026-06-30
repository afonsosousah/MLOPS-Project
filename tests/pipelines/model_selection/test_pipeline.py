import mlflow
import pandas as pd
import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner
from optuna.trial import FixedTrial
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from mlops_project.pipelines.model_selection import create_pipeline
from mlops_project.pipelines.model_selection.nodes import (
    build_regressor,
    model_selection,
    sample_search_space,
)

TEST_OPTUNA_TRIALS = 3
TEST_EXPERIMENT_NAME = "test_green_taxi_model_selection"

MODEL_SELECTION_PARAMS = {
    "enabled": True,
    "metric": "rmse",
    "search": {
        "enabled": True,
        "n_trials": TEST_OPTUNA_TRIALS,
        "direction": "minimize",
        "sampler": "tpe",
        "random_state": 42,
    },
    "candidates": {
        "dummy": {
            "type": "dummy",
            "eligible": False,
            "params": {"strategy": "mean"},
        },
        "ridge": {
            "type": "ridge",
            "eligible": True,
            "params": {"alpha": 1.0},
            "search_space": {
                "alpha": {"type": "categorical", "choices": [0.1, 1.0]},
            },
        },
    },
}


def _X() -> pd.DataFrame:
    return pd.DataFrame({"feature_a": [1.0, 2.0, 3.0, 4.0], "feature_b": [0, 1, 0, 1]})


def _y() -> pd.DataFrame:
    return pd.DataFrame({"tip_amount": [1.0, 2.0, 3.0, 4.0]})


def _mlflow_tracking_uri(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def test_model_selection_returns_selected_metadata(tmp_path) -> None:
    metadata = model_selection(
        _X(),
        _X(),
        _y(),
        _y(),
        ["feature_a", "feature_b"],
        MODEL_SELECTION_PARAMS,
        _mlflow_tracking_uri(tmp_path),
        TEST_EXPERIMENT_NAME,
    )

    assert metadata["selected_model_name"] == "ridge"
    assert metadata["selected_model_type"] == "ridge"
    assert "selected_model_params" in metadata
    assert metadata["selection_source"] == "optuna"
    assert metadata["selected_columns"] == ["feature_a", "feature_b"]
    assert metadata["selection_metric"] == "rmse"
    assert {candidate["name"] for candidate in metadata["fixed_candidates"]} == {
        "dummy",
        "ridge",
    }
    assert len(metadata["optuna_trials"]) == TEST_OPTUNA_TRIALS
    assert metadata["optuna_parent_run_id"]
    assert metadata["optuna_best_trial"]["model_name"] == "ridge"
    assert metadata["optuna_best_trial"]["mlflow_run_id"]
    assert metadata["optuna_best_trial"]["model_artifact_logged"] is True
    assert "rmse" in metadata["optuna_best_trial"]["train_metrics"]
    assert "rmse" in metadata["optuna_best_trial"]["validation_metrics"]
    assert all(trial["mlflow_run_id"] for trial in metadata["optuna_trials"])


def test_model_selection_keeps_dummy_reference_ineligible(tmp_path) -> None:
    metadata = model_selection(
        _X(),
        _X(),
        _y(),
        _y(),
        ["feature_a", "feature_b"],
        MODEL_SELECTION_PARAMS,
        _mlflow_tracking_uri(tmp_path),
        TEST_EXPERIMENT_NAME,
    )

    dummy_result = next(
        candidate
        for candidate in metadata["fixed_candidates"]
        if candidate["name"] == "dummy"
    )

    assert dummy_result["eligible"] is False
    assert dummy_result["metrics"]["rmse"] >= 0
    assert dummy_result["train_metrics"]["rmse"] >= 0
    assert dummy_result["validation_metrics"]["rmse"] >= 0
    assert metadata["selected_model_name"] != "dummy"
    assert {
        trial["model_name"]
        for trial in metadata["optuna_trials"]
        if trial["state"] == "COMPLETE"
    } == {"ridge"}
    assert (
        sum(trial["model_artifact_logged"] for trial in metadata["optuna_trials"]) == 1
    )


def test_model_selection_logs_trials_inside_active_mlflow_run(tmp_path) -> None:
    tracking_uri = _mlflow_tracking_uri(tmp_path)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(TEST_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="outer_kedro_style_run"):
        metadata = model_selection(
            _X(),
            _X(),
            _y(),
            _y(),
            ["feature_a", "feature_b"],
            MODEL_SELECTION_PARAMS,
            tracking_uri,
            TEST_EXPERIMENT_NAME,
        )

    assert metadata["optuna_parent_run_id"]
    assert metadata["optuna_best_trial"]["model_artifact_logged"] is True
    assert "train_metrics" in metadata["optuna_best_trial"]
    assert "validation_metrics" in metadata["optuna_best_trial"]


def test_build_regressor_supports_dummy_candidate() -> None:
    model = build_regressor(
        {"type": "dummy", "eligible": False, "params": {"strategy": "mean"}}
    )

    assert isinstance(model, DummyRegressor)


def test_build_regressor_supports_linear_regression_candidate() -> None:
    model = build_regressor(
        {
            "type": "linear_regression",
            "eligible": True,
            "params": {"fit_intercept": True},
        }
    )

    assert isinstance(model, LinearRegression)


def test_build_regressor_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported model_selection candidate type"):
        build_regressor({"type": "not_a_model", "params": {}})


def test_sample_search_space_supports_configured_types() -> None:
    trial = FixedTrial(
        {
            "tree__depth": 4,
            "tree__rate": 0.1,
            "tree__criterion": "squared_error",
        }
    )

    params = sample_search_space(
        trial,
        {
            "depth": {"type": "int", "low": 2, "high": 6},
            "rate": {"type": "float", "low": 0.01, "high": 0.2},
            "criterion": {
                "type": "categorical",
                "choices": ["squared_error", "absolute_error"],
            },
        },
        prefix="tree",
    )

    assert params == {
        "depth": 4,
        "rate": 0.1,
        "criterion": "squared_error",
    }


def test_sample_search_space_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported Optuna search-space type"):
        sample_search_space(
            FixedTrial({"bad": 1}),
            {"bad": {"type": "unsupported", "low": 1, "high": 2}},
        )


def test_model_selection_pipeline_creates_metadata(tmp_path) -> None:
    catalog = DataCatalog(
        {
            "X_train_preprocessed": MemoryDataset(data=_X()),
            "X_val_preprocessed": MemoryDataset(data=_X()),
            "y_train_data": MemoryDataset(data=_y()),
            "y_val_data": MemoryDataset(data=_y()),
            "best_columns": MemoryDataset(data=["feature_a", "feature_b"]),
            "params:model_selection": MemoryDataset(data=MODEL_SELECTION_PARAMS),
            "params:mlflow_tracking_uri": MemoryDataset(
                data=_mlflow_tracking_uri(tmp_path)
            ),
            "params:mlflow_experiment_name": MemoryDataset(data=TEST_EXPERIMENT_NAME),
            "selected_model_metadata": MemoryDataset(),
        }
    )

    SequentialRunner().run(create_pipeline(), catalog)

    assert catalog.load("selected_model_metadata")["selected_model_name"] == "ridge"
    assert catalog.load("selected_model_metadata")["selection_source"] == "optuna"
