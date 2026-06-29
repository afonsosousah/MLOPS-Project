import pandas as pd
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from mlops_project.pipelines.feature_selection import create_pipeline
from mlops_project.pipelines.feature_selection.nodes import select_features

FEATURE_SELECTION_PARAMS = {"enabled": False}


def _X_train() -> pd.DataFrame:
    return pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [0.0, 1.0]})


def _y_train() -> pd.DataFrame:
    return pd.DataFrame({"tip_amount": [2.0, 3.0]})


def test_select_features_returns_all_columns_when_disabled() -> None:
    selected = select_features(_X_train(), _y_train(), FEATURE_SELECTION_PARAMS)

    assert selected == ["feature_a", "feature_b"]


def test_feature_selection_pipeline_creates_best_columns() -> None:
    catalog = DataCatalog(
        {
            "X_train_preprocessed": MemoryDataset(data=_X_train()),
            "y_train_data": MemoryDataset(data=_y_train()),
            "params:feature_selection": MemoryDataset(data=FEATURE_SELECTION_PARAMS),
            "best_columns": MemoryDataset(),
        }
    )

    SequentialRunner().run(create_pipeline(), catalog)

    assert catalog.load("best_columns") == ["feature_a", "feature_b"]
