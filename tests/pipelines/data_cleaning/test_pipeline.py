import pandas as pd
import pytest

from mlops_project.pipelines.data_cleaning.nodes import data_cleaning

FLOAT_TOLERANCE = 0.1


BASE_ROW = {
    "VendorID": 2,
    "lpep_pickup_datetime": "2024-03-15 08:00:00",
    "lpep_dropoff_datetime": "2024-03-15 08:20:00",
    "store_and_fwd_flag": "N",
    "RatecodeID": 1.0,
    "PULocationID": 100,
    "DOLocationID": 150,
    "passenger_count": 1.0,
    "trip_distance": 3.5,
    "fare_amount": 15.0,
    "extra": 0.0,
    "mta_tax": 0.5,
    "tip_amount": 3.0,
    "tolls_amount": 0.0,
    "improvement_surcharge": 0.3,
    "total_amount": 18.8,
    "payment_type": 1,
    "trip_type": 1,
    "congestion_surcharge": 0.0,
    "source_partition": "2024/01",
    "source_year": 2024,
    "PU_borough": "Manhattan",
    "DO_borough": "Brooklyn",
}

PARAMETERS = {
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


@pytest.fixture
def raw_trip():
    return pd.DataFrame([BASE_ROW])


def test_data_cleaning_returns_dataframe(raw_trip):
    result = data_cleaning(raw_trip, PARAMETERS)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_data_cleaning_adds_engineered_features(raw_trip):
    result = data_cleaning(raw_trip, PARAMETERS)
    for col in [
        "trip_duration_min",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "is_weekend",
        "is_rush_hour",
        "is_night",
        "is_airport",
        "trip_id",
    ]:
        assert col in result.columns, (
            f"Expected engineered column '{col}' not in result"
        )


def test_data_cleaning_filters_non_credit_card():
    row = {**BASE_ROW, "payment_type": 2}  # cash — should be filtered
    result = data_cleaning(pd.DataFrame([row]), PARAMETERS)
    assert len(result) == 0


def test_data_cleaning_filters_outlier_distance():
    row = {**BASE_ROW, "trip_distance": 999.0}  # above max_trip_distance=100
    result = data_cleaning(pd.DataFrame([row]), PARAMETERS)
    assert len(result) == 0


def test_data_cleaning_filters_outlier_fare():
    row = {**BASE_ROW, "fare_amount": 9999.0}  # above max_fare_amount=500
    result = data_cleaning(pd.DataFrame([row]), PARAMETERS)
    assert len(result) == 0


def test_data_cleaning_filters_short_trips():
    row = {**BASE_ROW, "lpep_dropoff_datetime": "2024-03-15 08:00:30"}  # 30 sec < 1 min
    result = data_cleaning(pd.DataFrame([row]), PARAMETERS)
    assert len(result) == 0


def test_data_cleaning_trip_duration_correct(raw_trip):
    result = data_cleaning(raw_trip, PARAMETERS)
    assert abs(result["trip_duration_min"].iloc[0] - 20.0) < FLOAT_TOLERANCE
