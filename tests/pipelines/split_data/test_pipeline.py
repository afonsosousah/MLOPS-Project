import pandas as pd
import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from mlops_project.pipelines.split_data import create_pipeline
from mlops_project.pipelines.split_data.nodes import split_data

FEATURE_STORE_PARAMS = {"use_feature_store": False}


def _ingested_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "VendorID": [2, 2],
            "lpep_pickup_datetime": [
                "2024-06-01 08:00:00",
                "2025-08-01 08:00:00",
            ],
            "lpep_dropoff_datetime": [
                "2024-06-01 08:15:00",
                "2025-08-01 08:20:00",
            ],
            "store_and_fwd_flag": ["N", "N"],
            "RatecodeID": [1, 1],
            "PULocationID": [7, 41],
            "DOLocationID": [42, 74],
            "passenger_count": [1, 1],
            "trip_distance": [1.2, 2.5],
            "fare_amount": [8.0, 12.5],
            "extra": [1.0, 1.0],
            "mta_tax": [0.5, 0.5],
            "tip_amount": [2.0, 3.0],
            "tolls_amount": [0.0, 0.0],
            "improvement_surcharge": [1.0, 1.0],
            "total_amount": [12.5, 17.0],
            "payment_type": [1, 1],
            "trip_type": [1, 1],
            "congestion_surcharge": [2.75, 2.75],
            "source_partition": ["2024-06", "2025-08"],
            "source_year": [2024, 2025],
            "PU_borough": ["Queens", "Manhattan"],
            "DO_borough": ["Brooklyn", "Queens"],
        }
    )


def test_split_data_creates_chronological_train_and_test() -> None:
    train_data, test_data = split_data(_ingested_data(), "2025-07-01")

    assert train_data["tip_amount"].tolist() == [2.0]
    assert test_data["tip_amount"].tolist() == [3.0]
    assert train_data.index.tolist() == [0]
    assert test_data.index.tolist() == [0]


def test_split_data_raises_for_empty_side() -> None:
    with pytest.raises(ValueError, match="empty train or test"):
        split_data(_ingested_data(), "2023-01-01")


def test_split_data_pipeline_creates_train_and_test_outputs() -> None:
    catalog = DataCatalog(
        {
            "data_clean": MemoryDataset(data=_ingested_data()),
            "params:feature_store": MemoryDataset(data=FEATURE_STORE_PARAMS),
            "params:train_test_split_date": MemoryDataset(data="2025-07-01"),
            "train_data": MemoryDataset(),
            "test_data": MemoryDataset(),
        }
    )

    SequentialRunner().run(create_pipeline(), catalog)

    assert catalog.load("train_data")["tip_amount"].tolist() == [2.0]
    assert catalog.load("test_data")["tip_amount"].tolist() == [3.0]
