import pandas as pd
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from mlops_project.pipelines.preprocessing_train import create_pipeline
from mlops_project.pipelines.preprocessing_train.nodes import preprocessing_train

PREPROCESSING_PARAMS = {
    "columns_to_drop": [
        "VendorID",
        "lpep_pickup_datetime",
        "lpep_dropoff_datetime",
        "store_and_fwd_flag",
        "payment_type",
        "total_amount",
        "source_partition",
        "trip_id",
    ]
}


def _features(months: list[str], boroughs: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "VendorID": [2, 2],
            "lpep_pickup_datetime": [
                f"{months[0]}-01 08:00:00",
                f"{months[1]}-01 18:00:00",
            ],
            "lpep_dropoff_datetime": [
                f"{months[0]}-01 08:15:00",
                f"{months[1]}-01 18:20:00",
            ],
            "store_and_fwd_flag": ["N", "N"],
            "RatecodeID": [1, 2],
            "PULocationID": [7, 132],
            "DOLocationID": [42, 74],
            "passenger_count": [1, 2],
            "trip_distance": [1.2, 2.5],
            "fare_amount": [8.0, 12.5],
            "extra": [1.0, 1.0],
            "mta_tax": [0.5, 0.5],
            "tolls_amount": [0.0, 0.0],
            "improvement_surcharge": [1.0, 1.0],
            "total_amount": [12.5, 17.0],
            "payment_type": [1, 1],
            "trip_type": [1, 1],
            "congestion_surcharge": [2.75, 2.75],
            "source_partition": months,
            "source_year": [2024, 2025],
            "PU_borough": boroughs,
            "DO_borough": ["Brooklyn", "Queens"],
        }
    )


def test_preprocessing_train_fits_on_train_and_transforms_validation() -> None:
    result = preprocessing_train(
        _features(["2024-06", "2024-07"], ["Queens", "Manhattan"]),
        _features(["2025-01", "2025-02"], ["Queens", "Unknown"]),
        PREPROCESSING_PARAMS,
        "tip_amount",
    )

    X_train_preprocessed, X_val_preprocessed, transformer, columns, report = result
    assert X_train_preprocessed.columns.tolist() == columns
    assert X_val_preprocessed.columns.tolist() == columns
    assert transformer.output_columns_ == columns
    assert report["production_column_count"] == len(columns)


def test_preprocessing_train_pipeline_creates_expected_outputs() -> None:
    catalog = DataCatalog(
        {
            "X_train_data": MemoryDataset(
                data=_features(["2024-06", "2024-07"], ["Queens", "Manhattan"])
            ),
            "X_val_data": MemoryDataset(
                data=_features(["2025-01", "2025-02"], ["Queens", "Unknown"])
            ),
            "params:preprocessing": MemoryDataset(data=PREPROCESSING_PARAMS),
            "params:target_column": MemoryDataset(data="tip_amount"),
            "X_train_preprocessed": MemoryDataset(),
            "X_val_preprocessed": MemoryDataset(),
            "preprocessing_transformer": MemoryDataset(),
            "production_columns": MemoryDataset(),
            "reporting_data_train": MemoryDataset(),
        }
    )

    SequentialRunner().run(create_pipeline(), catalog)

    assert catalog.load("X_train_preprocessed").columns.tolist() == catalog.load(
        "production_columns"
    )
