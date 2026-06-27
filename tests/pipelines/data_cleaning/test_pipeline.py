import pandas as pd
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from mlops_project.pipelines.data_cleaning import create_pipeline
from mlops_project.pipelines.data_cleaning.nodes import clean_data


DATA_CLEANING_PARAMS = {
    "min_pickup_datetime": "2024-01-01",
    "min_trip_distance": 0,
    "max_trip_distance": 100,
    "min_fare_amount": 0,
    "max_fare_amount": 500,
    "min_duration_min": 1,
    "max_duration_min": 180,
    "min_tip_amount": 0,
    "max_tip_amount": 200,
    "min_location_id": 1,
    "max_location_id": 263,
}


def _base_valid_row(row_id: str = "valid") -> dict:
    return {
        "row_id": row_id,
        "lpep_pickup_datetime": "2024-01-01 08:00:00",
        "lpep_dropoff_datetime": "2024-01-01 08:20:00",
        "PULocationID": 7,
        "DOLocationID": 42,
        "trip_distance": 2.5,
        "fare_amount": 12.0,
        "tip_amount": 3.0,
        "trip_type": 1,
        "payment_type": 1,
        "ehail_fee": None,
        "cbd_congestion_fee": 2.75,
    }


def _dirty_modeling_data() -> pd.DataFrame:
    rows = [_base_valid_row(), _base_valid_row()]

    cash_payment = _base_valid_row("cash_payment")
    cash_payment["payment_type"] = 2
    rows.append(cash_payment)

    missing_required = _base_valid_row("missing_required")
    missing_required["trip_type"] = None
    rows.append(missing_required)

    invalid_pickup_location = _base_valid_row("invalid_pickup_location")
    invalid_pickup_location["PULocationID"] = 999
    rows.append(invalid_pickup_location)

    invalid_dropoff_location = _base_valid_row("invalid_dropoff_location")
    invalid_dropoff_location["DOLocationID"] = 0
    rows.append(invalid_dropoff_location)

    invalid_distance = _base_valid_row("invalid_distance")
    invalid_distance["trip_distance"] = 0
    rows.append(invalid_distance)

    invalid_fare = _base_valid_row("invalid_fare")
    invalid_fare["fare_amount"] = 0
    rows.append(invalid_fare)

    invalid_tip = _base_valid_row("invalid_tip")
    invalid_tip["tip_amount"] = 200
    rows.append(invalid_tip)

    invalid_datetime_order = _base_valid_row("invalid_datetime_order")
    invalid_datetime_order["lpep_dropoff_datetime"] = "2024-01-01 07:59:00"
    rows.append(invalid_datetime_order)

    too_short_duration = _base_valid_row("too_short_duration")
    too_short_duration["lpep_dropoff_datetime"] = "2024-01-01 08:00:30"
    rows.append(too_short_duration)

    return pd.DataFrame(rows)


def test_clean_data_filters_invalid_rows_and_drops_transient_columns() -> None:
    cleaned_data = clean_data(_dirty_modeling_data(), DATA_CLEANING_PARAMS)

    assert cleaned_data["row_id"].tolist() == ["valid"]
    assert "ehail_fee" not in cleaned_data.columns
    assert "cbd_congestion_fee" not in cleaned_data.columns
    assert cleaned_data.index.tolist() == [0]
    assert pd.api.types.is_datetime64_any_dtype(cleaned_data["lpep_pickup_datetime"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned_data["lpep_dropoff_datetime"])
    assert cleaned_data["payment_type"].eq(1).all()
    assert cleaned_data["trip_distance"].between(0, 100, inclusive="neither").all()
    assert cleaned_data["fare_amount"].between(0, 500, inclusive="neither").all()
    assert cleaned_data["tip_amount"].between(0, 200, inclusive="left").all()
    assert cleaned_data.duplicated().sum() == 0


def test_data_cleaning_pipeline_creates_cleaned_modeling_data() -> None:
    catalog = DataCatalog(
        {
            "modeling_ingested_data": MemoryDataset(data=_dirty_modeling_data()),
            "params:data_cleaning": MemoryDataset(data=DATA_CLEANING_PARAMS),
            "modeling_cleaned_data": MemoryDataset(),
        }
    )

    SequentialRunner().run(create_pipeline(), catalog)

    cleaned_data = catalog.load("modeling_cleaned_data")

    assert cleaned_data["row_id"].tolist() == ["valid"]
    assert "ehail_fee" not in cleaned_data.columns
    assert "cbd_congestion_fee" not in cleaned_data.columns
