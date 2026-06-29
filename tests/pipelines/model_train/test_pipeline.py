import pandas as pd
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from mlops_project.pipelines.model_train import create_pipeline
from mlops_project.pipelines.model_train.nodes import model_train


def _X() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [0.0, 1.0, 0.0, 1.0],
        }
    )


def _y() -> pd.DataFrame:
    return pd.DataFrame({"tip_amount": [1.0, 2.0, 3.0, 4.0]})


def _model_train_params(tmp_path):
    return {
        "target_col": "tip_amount",
        "mlflow_tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "mlflow_experiment_name": "test_green_taxi",
        "selected_model_name": "small_rf",
        "model_params": {
            "n_estimators": 5,
            "max_depth": 3,
            "random_state": 42,
            "n_jobs": 1,
        },
    }


MODEL_SELECTION_PARAMS = {
    "candidates": {
        "small_rf": {
            "n_estimators": 5,
            "max_depth": 3,
            "random_state": 42,
            "n_jobs": 1,
        }
    }
}


def test_model_train_consumes_preprocessed_features(tmp_path) -> None:
    model, metrics, predictions, plot = model_train(
        _X(),
        _X(),
        _y(),
        _y(),
        ["feature_a", "feature_b"],
        {"selected_model_name": "small_rf"},
        _model_train_params(tmp_path),
        MODEL_SELECTION_PARAMS,
    )

    assert hasattr(model, "predict")
    assert "rmse" in metrics
    assert predictions.columns.tolist() == [
        "actual_tip_amount",
        "predicted_tip_amount",
        "residual",
    ]
    assert plot is not None


def test_model_train_pipeline_creates_outputs(tmp_path) -> None:
    catalog = DataCatalog(
        {
            "X_train_preprocessed": MemoryDataset(data=_X()),
            "X_val_preprocessed": MemoryDataset(data=_X()),
            "y_train_data": MemoryDataset(data=_y()),
            "y_val_data": MemoryDataset(data=_y()),
            "best_columns": MemoryDataset(data=["feature_a", "feature_b"]),
            "selected_model_metadata": MemoryDataset(
                data={"selected_model_name": "small_rf"}
            ),
            "params:model_train": MemoryDataset(data=_model_train_params(tmp_path)),
            "params:model_selection": MemoryDataset(data=MODEL_SELECTION_PARAMS),
            "production_model": MemoryDataset(),
            "production_model_metrics": MemoryDataset(),
            "validation_predictions": MemoryDataset(),
            "output_plot": MemoryDataset(),
        }
    )

    SequentialRunner().run(create_pipeline(), catalog)

    assert "rmse" in catalog.load("production_model_metrics")
    assert not catalog.load("validation_predictions").empty
