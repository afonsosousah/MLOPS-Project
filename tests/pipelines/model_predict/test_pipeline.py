import numpy as np
import pandas as pd
import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from mlops_project.pipelines.model_predict import create_pipeline
from mlops_project.pipelines.model_predict.nodes import model_predict

EXPECTED_ROWS = 2
EXPECTED_PREDICTION_MEAN = 3.0


class DummyModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features["feature_a"].to_numpy() * 2


def _batch_data() -> pd.DataFrame:
    return pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [10.0, 20.0]})


def test_model_predict_selects_production_columns_and_adds_predictions() -> None:
    predictions, summary = model_predict(
        data=_batch_data(),
        model=DummyModel(),
        columns=["feature_a"],
    )

    assert predictions["predicted_tip_amount"].tolist() == [2.0, 4.0]
    assert summary["rows"] == EXPECTED_ROWS
    assert summary["prediction_mean"] == EXPECTED_PREDICTION_MEAN


def test_model_predict_raises_when_no_production_columns_exist() -> None:
    with pytest.raises(ValueError, match="No production model columns"):
        model_predict(
            data=_batch_data(),
            model=DummyModel(),
            columns=["missing_feature"],
        )


def test_model_predict_pipeline_creates_predictions_and_summary() -> None:
    catalog = DataCatalog(
        {
            "X_batch_preprocessed": MemoryDataset(data=_batch_data()),
            "production_model": MemoryDataset(data=DummyModel()),
            "production_columns": MemoryDataset(data=["feature_a"]),
            "predictions": MemoryDataset(),
            "predict_describe": MemoryDataset(),
        }
    )

    SequentialRunner().run(create_pipeline(), catalog)

    predictions = catalog.load("predictions")
    summary = catalog.load("predict_describe")
    assert predictions["predicted_tip_amount"].tolist() == [2.0, 4.0]
    assert summary["rows"] == EXPECTED_ROWS
