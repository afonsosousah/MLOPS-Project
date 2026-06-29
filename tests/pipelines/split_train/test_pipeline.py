import pandas as pd
import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from mlops_project.pipelines.split_train import create_pipeline
from mlops_project.pipelines.split_train.nodes import split_train


def _train_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lpep_pickup_datetime": [
                "2024-06-01 08:00:00",
                "2025-03-01 08:00:00",
            ],
            "feature_a": [1.0, 2.0],
            "tip_amount": [2.0, 3.0],
        }
    )


def test_split_train_creates_chronological_x_y_sets() -> None:
    X_train, X_val, y_train, y_val = split_train(
        _train_data(),
        "2025-01-01",
        "tip_amount",
    )

    assert X_train["feature_a"].tolist() == [1.0]
    assert X_val["feature_a"].tolist() == [2.0]
    assert y_train["tip_amount"].tolist() == [2.0]
    assert y_val["tip_amount"].tolist() == [3.0]
    assert "tip_amount" not in X_train.columns


def test_split_train_raises_for_empty_side() -> None:
    with pytest.raises(ValueError, match="empty split"):
        split_train(_train_data(), "2023-01-01", "tip_amount")


def test_split_train_pipeline_creates_outputs() -> None:
    catalog = DataCatalog(
        {
            "train_data": MemoryDataset(data=_train_data()),
            "params:train_val_split_date": MemoryDataset(data="2025-01-01"),
            "params:target_column": MemoryDataset(data="tip_amount"),
            "X_train_data": MemoryDataset(),
            "X_val_data": MemoryDataset(),
            "y_train_data": MemoryDataset(),
            "y_val_data": MemoryDataset(),
        }
    )

    SequentialRunner().run(create_pipeline(), catalog)

    assert catalog.load("X_train_data")["feature_a"].tolist() == [1.0]
    assert catalog.load("X_val_data")["feature_a"].tolist() == [2.0]
