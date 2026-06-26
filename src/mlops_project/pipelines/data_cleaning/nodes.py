import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def clean_green_taxi_data(
    ingested_data: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Clean full Green Taxi dataset before feature engineering / Hopsworks upload."""
    n_start = len(ingested_data)
    df = ingested_data.copy()

    min_pickup_datetime = parameters.get("min_pickup_datetime", "2024-01-01")
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

    df = df.drop(columns=["ehail_fee", "cbd_congestion_fee"], errors="ignore")

    df["lpep_pickup_datetime"] = pd.to_datetime(
        df["lpep_pickup_datetime"], errors="coerce"
    )
    df["lpep_dropoff_datetime"] = pd.to_datetime(
        df["lpep_dropoff_datetime"], errors="coerce"
    )

    # Credit card only: cash tips are not reliably recorded.
    df = df[df["payment_type"].eq(1)].copy()


    #Missing Values
    df = df.dropna(
        subset=[
            "lpep_pickup_datetime",
            "lpep_dropoff_datetime",
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "fare_amount",
            "tip_amount",
            "trip_type",
        ]
    )

    df = df[df["PULocationID"].between(min_location_id, max_location_id)]
    df = df[df["DOLocationID"].between(min_location_id, max_location_id)]

    df = df[
        df["trip_distance"].gt(min_trip_distance)
        & df["trip_distance"].lt(max_trip_distance)
    ]

    df = df[
        df["fare_amount"].gt(min_fare_amount)
        & df["fare_amount"].lt(max_fare_amount)
    ]

    df = df[df["lpep_pickup_datetime"] >= min_pickup_datetime]
    df = df[df["lpep_dropoff_datetime"] >(df["lpep_pickup_datetime"])]

    duration_min = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60

    df = df[(duration_min >= min_duration_min) & (duration_min <= max_duration_min)]

    df = df[
        df["tip_amount"].ge(min_tip_amount)
        & df["tip_amount"].lt(max_tip_amount)
    ]

    df = df.drop_duplicates().reset_index(drop=True)

    logger.info(
        "Cleaned Green Taxi data: %d -> %d rows retained (%.2f%%).",
        n_start,
        len(df),
        len(df) / n_start * 100 if n_start else 0,
    )

    return df

