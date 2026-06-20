from pathlib import Path

import pandas as pd
import pytest

from mlops_project.profiling import (
    add_trip_derived_columns,
    parse_green_taxi_month,
    quality_check_summary,
    serving_time_column_classification,
)


def test_parse_green_taxi_month_from_path():
    parsed = parse_green_taxi_month(Path("data/01_raw/green_taxi/2025/green_tripdata_2025-04.parquet"))

    assert parsed == {"year": 2025, "month": 4, "period": "2025-04"}


def test_parse_green_taxi_month_rejects_unexpected_name():
    with pytest.raises(ValueError, match="Not a Green Taxi"):
        parse_green_taxi_month("yellow_tripdata_2025-04.parquet")


def test_quality_check_summary_counts_invalid_duration_and_distance():
    frame = pd.DataFrame(
        {
            "lpep_pickup_datetime": pd.to_datetime(["2025-01-01 10:00:00", "2025-01-01 11:00:00"]),
            "lpep_dropoff_datetime": pd.to_datetime(["2025-01-01 10:10:00", "2025-01-01 10:55:00"]),
            "trip_distance": [1.2, -0.3],
            "fare_amount": [8.0, -5.0],
            "total_amount": [10.0, -6.0],
            "PULocationID": [74, None],
            "DOLocationID": [75, 76],
            "passenger_count": [1, 0],
        }
    )

    summary = quality_check_summary(frame).set_index("check")

    assert summary.loc["non_positive_duration", "rows"] == 1
    assert summary.loc["negative_trip_distance", "rows"] == 1
    assert summary.loc["negative_fare_amount", "rows"] == 1
    assert summary.loc["missing_pickup_location", "rows"] == 1
    assert summary.loc["passenger_count_zero_or_negative", "rows"] == 1


def test_add_trip_derived_columns_computes_minutes():
    frame = pd.DataFrame(
        {
            "lpep_pickup_datetime": ["2025-01-01 10:00:00"],
            "lpep_dropoff_datetime": ["2025-01-01 10:15:00"],
        }
    )

    result = add_trip_derived_columns(frame)

    assert result.loc[0, "trip_duration_minutes"] == 15
    assert result.loc[0, "pickup_hour"] == 10


def test_serving_time_classification_marks_post_trip_columns():
    result = serving_time_column_classification(["PULocationID", "total_amount", "VendorID"]).set_index("column")

    assert result.loc["PULocationID", "availability"] == "serving_candidate"
    assert result.loc["total_amount", "availability"] == "post_trip_only"
    assert result.loc["VendorID", "availability"] == "needs_review"
