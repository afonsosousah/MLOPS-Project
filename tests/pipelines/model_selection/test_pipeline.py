import pandas as pd
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from mlops_project.pipelines.model_selection import create_pipeline
from mlops_project.pipelines.model_selection.nodes import model_selection

MODEL_SELECTION_PARAMS = {
    "enabled": True,
    "max_rows": 0,
    "candidates": {
        "small_rf": {
            "n_estimators": 5,
            "max_depth": 3,
            "random_state": 42,
            "n_jobs": 1,
        }
    },
}


def _X() -> pd.DataFrame:
    return pd.DataFrame({"feature_a": [1.0, 2.0, 3.0, 4.0], "feature_b": [0, 1, 0, 1]})


def _y() -> pd.DataFrame:
    return pd.DataFrame({"tip_amount": [1.0, 2.0, 3.0, 4.0]})


def test_model_selection_returns_selected_metadata() -> None:
    metadata = model_selection(
        _X(),
        _X(),
        _y(),
        _y(),
        ["feature_a", "feature_b"],
        MODEL_SELECTION_PARAMS,
    )

    assert metadata["selected_model_name"] == "small_rf"
    assert metadata["selected_columns"] == ["feature_a", "feature_b"]


def test_model_selection_pipeline_creates_metadata() -> None:
    catalog = DataCatalog(
        {
            "X_train_preprocessed": MemoryDataset(data=_X()),
            "X_val_preprocessed": MemoryDataset(data=_X()),
            "y_train_data": MemoryDataset(data=_y()),
            "y_val_data": MemoryDataset(data=_y()),
            "best_columns": MemoryDataset(data=["feature_a", "feature_b"]),
            "params:model_selection": MemoryDataset(data=MODEL_SELECTION_PARAMS),
            "selected_model_metadata": MemoryDataset(),
        }
    )

    SequentialRunner().run(create_pipeline(), catalog)

    assert catalog.load("selected_model_metadata")["selected_model_name"] == "small_rf"
