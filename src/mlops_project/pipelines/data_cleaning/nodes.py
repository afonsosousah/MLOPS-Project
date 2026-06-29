import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

TRANSIENT_COLUMNS = ["ehail_fee", "cbd_congestion_fee"]
DATETIME_COLUMNS = ["lpep_pickup_datetime", "lpep_dropoff_datetime"]
REQUIRED_COLUMNS = [
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "trip_type",
]

def _coerce_datetime_columns(data: pd.DataFrame) -> pd.DataFrame:
    coerced = data.copy()
    for column in DATETIME_COLUMNS:
        coerced[column] = pd.to_datetime(coerced[column], errors="coerce")
    return coerced



def data_cleaning(
    ingested_data: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Apply deterministic row filters before chronological splitting."""
    rows_before = len(ingested_data)
    data = ingested_data.copy()

    min_pickup_datetime = pd.Timestamp(
        parameters.get("min_pickup_datetime", "2024-01-01")
    )
    min_trip_distance = parameters.get("min_trip_distance", 0)
    max_trip_distance = parameters.get("max_trip_distance", 100)
    min_fare_amount = parameters.get("min_fare_amount", 0)
    max_fare_amount = parameters.get("max_fare_amount", 500)
    min_duration_min = parameters.get("min_duration_min", 1)
    max_duration_min = parameters.get("max_duration_min", 180)
    min_tip_amount = parameters.get("min_tip_amount", 0)
    max_tip_amount = parameters.get("max_tip_amount", 200)
    min_location_id = parameters.get("min_location_id", 1)
    max_location_id = parameters.get("max_location_id", 263)

    data = data.drop(columns=TRANSIENT_COLUMNS, errors="ignore")
    data = _coerce_datetime_columns(data)
    data = data[data["payment_type"].eq(1)].copy()
    data = data.dropna(subset=REQUIRED_COLUMNS)

    data = data[data["PULocationID"].between(min_location_id, max_location_id)]
    data = data[data["DOLocationID"].between(min_location_id, max_location_id)]
    data = data[
        data["trip_distance"].gt(min_trip_distance)
        & data["trip_distance"].lt(max_trip_distance)
    ]
    data = data[
        data["fare_amount"].gt(min_fare_amount)
        & data["fare_amount"].lt(max_fare_amount)
    ]
    data = data[data["lpep_pickup_datetime"] >= min_pickup_datetime]
    data = data[data["lpep_dropoff_datetime"] > data["lpep_pickup_datetime"]]

    duration_min = (
        data["lpep_dropoff_datetime"] - data["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    data = data[(duration_min >= min_duration_min) & (duration_min <= max_duration_min)]
    data = data[
        data["tip_amount"].ge(min_tip_amount) & data["tip_amount"].lt(max_tip_amount)
    ]
    data = data.drop_duplicates().reset_index(drop=True)

    logger.info(
        "Filtered ingested data before split: %d -> %d rows retained (%.2f%%).",
        rows_before,
        len(data),
        len(data) / rows_before * 100 if rows_before else 0,
    )
    return data