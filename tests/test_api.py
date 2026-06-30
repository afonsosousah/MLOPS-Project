import pandas as pd

from mlops_project.api import main
from mlops_project.api.main import TripFeatures

EXPECTED_PREDICTION = 2.5


class RecordingTransformer:
    def __init__(self) -> None:
        self.input_columns: list[str] = []

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.input_columns = data.columns.tolist()
        return pd.DataFrame({"feature_a": [1.0]})


class DummyModel:
    def predict(self, features: pd.DataFrame) -> list[float]:
        return [EXPECTED_PREDICTION]


def test_predict_preserves_borough_features_for_preprocessor(monkeypatch) -> None:
    transformer = RecordingTransformer()
    monkeypatch.setattr(main, "_model", DummyModel())
    monkeypatch.setattr(main, "_transformer", transformer)
    monkeypatch.setattr(main, "_columns", ["feature_a"])

    response = main.predict(
        TripFeatures(
            PULocationID=74,
            DOLocationID=42,
            PU_borough="Manhattan",
            DO_borough="Brooklyn",
        )
    )

    assert response.predicted_tip_amount == EXPECTED_PREDICTION
    assert "PU_borough" in transformer.input_columns
    assert "DO_borough" in transformer.input_columns
