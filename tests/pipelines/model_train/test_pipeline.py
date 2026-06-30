import pandas as pd
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from mlops_project.pipelines.model_train import create_pipeline
from mlops_project.pipelines.model_train import nodes as model_train_nodes
from mlops_project.pipelines.model_train.nodes import model_train

TUNED_RIDGE_ALPHA = 7.0
TEST_EXPERIMENT_NAME = "test_green_taxi_model_train"


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
        "selected_model_name": "ridge",
        "model_params": {
            "n_estimators": 5,
            "max_depth": 3,
            "random_state": 42,
            "n_jobs": 1,
        },
        "feature_importance_sample_size": 4,
        "feature_importance_max_display": 2,
        "shap_sample_size": 3,
        "shap_background_sample_size": 3,
        "shap_max_display": 2,
        "explainability_random_state": 42,
    }


def _mlflow_tracking_uri(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


MODEL_SELECTION_PARAMS = {
    "candidates": {
        "ridge": {
            "type": "ridge",
            "eligible": True,
            "params": {"alpha": 1.0},
        },
        "random_forest": {
            "type": "random_forest",
            "eligible": True,
            "params": {
                "n_estimators": 5,
                "max_depth": 3,
                "random_state": 42,
                "n_jobs": 1,
            },
        },
        "dummy": {
            "type": "dummy",
            "eligible": False,
            "params": {"strategy": "mean"},
        },
    }
}


def _redirect_explainability_paths(monkeypatch, tmp_path) -> None:
    reporting_dir = tmp_path / "reporting"
    monkeypatch.setattr(model_train_nodes, "REPORTING_DIR", reporting_dir)
    monkeypatch.setattr(
        model_train_nodes,
        "FEATURE_IMPORTANCE_PLOT_PATH",
        reporting_dir / "feature_importance.png",
    )
    monkeypatch.setattr(
        model_train_nodes,
        "FEATURE_IMPORTANCE_TABLE_PATH",
        reporting_dir / "feature_importance.csv",
    )
    monkeypatch.setattr(
        model_train_nodes,
        "SHAP_SUMMARY_PLOT_PATH",
        reporting_dir / "shap_summary.png",
    )
    monkeypatch.setattr(
        model_train_nodes,
        "SHAP_SUMMARY_TABLE_PATH",
        reporting_dir / "shap_summary.csv",
    )


def test_model_train_consumes_preprocessed_features(tmp_path, monkeypatch) -> None:
    _redirect_explainability_paths(monkeypatch, tmp_path)

    model, metrics, predictions, explainability_metadata = model_train(
        _X(),
        _X(),
        _y(),
        _y(),
        ["feature_a", "feature_b"],
        {
            "selected_model_name": "ridge",
            "selected_model_type": "ridge",
            "selected_model_params": {"alpha": TUNED_RIDGE_ALPHA},
        },
        _model_train_params(tmp_path),
        MODEL_SELECTION_PARAMS,
        _mlflow_tracking_uri(tmp_path),
        TEST_EXPERIMENT_NAME,
    )

    assert hasattr(model, "predict")
    assert model.__class__.__name__ == "Ridge"
    assert model.alpha == TUNED_RIDGE_ALPHA
    assert "rmse" in metrics
    assert "train_rmse" in metrics
    assert "validation_rmse" in metrics
    assert "rmse" in metrics["train_metrics"]
    assert "rmse" in metrics["validation_metrics"]
    assert predictions.columns.tolist() == [
        "actual_tip_amount",
        "predicted_tip_amount",
        "residual",
    ]
    assert explainability_metadata["feature_importance"]["generated"] is True
    assert explainability_metadata["feature_importance"]["method"] == "absolute_coef_"
    assert explainability_metadata["shap"]["generated"] is True
    assert explainability_metadata["shap"]["method"] == "linear_explainer"
    assert (tmp_path / "reporting" / "feature_importance.png").exists()
    assert (tmp_path / "reporting" / "shap_summary.png").exists()


def test_model_train_pipeline_creates_outputs(tmp_path, monkeypatch) -> None:
    _redirect_explainability_paths(monkeypatch, tmp_path)

    catalog = DataCatalog(
        {
            "X_train_preprocessed": MemoryDataset(data=_X()),
            "X_val_preprocessed": MemoryDataset(data=_X()),
            "y_train_data": MemoryDataset(data=_y()),
            "y_val_data": MemoryDataset(data=_y()),
            "best_columns": MemoryDataset(data=["feature_a", "feature_b"]),
            "selected_model_metadata": MemoryDataset(
                data={
                    "selected_model_name": "ridge",
                    "selected_model_type": "ridge",
                    "selected_model_params": {"alpha": TUNED_RIDGE_ALPHA},
                }
            ),
            "params:model_train": MemoryDataset(data=_model_train_params(tmp_path)),
            "params:model_selection": MemoryDataset(data=MODEL_SELECTION_PARAMS),
            "params:mlflow_tracking_uri": MemoryDataset(
                data=_mlflow_tracking_uri(tmp_path)
            ),
            "params:mlflow_experiment_name": MemoryDataset(data=TEST_EXPERIMENT_NAME),
            "production_model": MemoryDataset(),
            "production_model_metrics": MemoryDataset(),
            "validation_predictions": MemoryDataset(),
            "model_explainability_metadata": MemoryDataset(),
        }
    )

    SequentialRunner().run(create_pipeline(), catalog)

    production_metrics = catalog.load("production_model_metrics")
    assert "rmse" in production_metrics
    assert "train_rmse" in production_metrics
    assert "validation_rmse" in production_metrics
    assert catalog.load("production_model").alpha == TUNED_RIDGE_ALPHA
    assert not catalog.load("validation_predictions").empty
    assert catalog.load("model_explainability_metadata")["shap"]["generated"] is True


def test_model_train_writes_tree_importance_and_shap(tmp_path, monkeypatch) -> None:
    _redirect_explainability_paths(monkeypatch, tmp_path)

    _, _, _, explainability_metadata = model_train(
        _X(),
        _X(),
        _y(),
        _y(),
        ["feature_a", "feature_b"],
        {
            "selected_model_name": "random_forest",
            "selected_model_type": "random_forest",
            "selected_model_params": {
                "n_estimators": 5,
                "max_depth": 3,
                "random_state": 42,
                "n_jobs": 1,
            },
        },
        _model_train_params(tmp_path),
        MODEL_SELECTION_PARAMS,
        _mlflow_tracking_uri(tmp_path),
        TEST_EXPERIMENT_NAME,
    )

    assert explainability_metadata["feature_importance"]["generated"] is True
    assert (
        explainability_metadata["feature_importance"]["method"]
        == "feature_importances_"
    )
    assert explainability_metadata["shap"]["generated"] is True
    assert explainability_metadata["shap"]["method"] == "tree_explainer"
    assert (tmp_path / "reporting" / "feature_importance.csv").exists()
    assert (tmp_path / "reporting" / "shap_summary.csv").exists()


def test_model_train_skips_explainability_for_unsupported_model(
    tmp_path,
    monkeypatch,
) -> None:
    _redirect_explainability_paths(monkeypatch, tmp_path)

    model, _, _, explainability_metadata = model_train(
        _X(),
        _X(),
        _y(),
        _y(),
        ["feature_a", "feature_b"],
        {
            "selected_model_name": "dummy",
            "selected_model_type": "dummy",
            "selected_model_params": {"strategy": "mean"},
        },
        _model_train_params(tmp_path),
        MODEL_SELECTION_PARAMS,
        _mlflow_tracking_uri(tmp_path),
        TEST_EXPERIMENT_NAME,
    )

    assert model.__class__.__name__ == "DummyRegressor"
    assert explainability_metadata["feature_importance"]["generated"] is False
    assert explainability_metadata["shap"]["generated"] is False
    assert (
        "does not expose"
        in explainability_metadata["feature_importance"]["skip_reason"]
    )
    assert "not configured for SHAP" in explainability_metadata["shap"]["skip_reason"]
    assert not (tmp_path / "reporting" / "feature_importance.png").exists()
    assert not (tmp_path / "reporting" / "shap_summary.png").exists()
