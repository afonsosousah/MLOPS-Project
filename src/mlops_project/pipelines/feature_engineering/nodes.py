import pandas as pd
import logging
import os
import hopsworks
from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Airport zone IDs (JFK, Newark, LaGuardia)
AIRPORT_ZONE_IDS = [1, 132, 138]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer new features from taxi trip data."""

    df = df.copy()
    n_cols_before = len(df.columns)

    # -------------------------
    # Temporal features
    # -------------------------
    pickup = pd.to_datetime(df["lpep_pickup_datetime"])
    dropoff = pd.to_datetime(df["lpep_dropoff_datetime"])

    df["trip_duration_min"] = (
        (dropoff - pickup).dt.total_seconds() / 60
    )

    df["pickup_hour"] = pickup.dt.hour
    df["pickup_dayofweek"] = pickup.dt.dayofweek
    df["pickup_month"] = pickup.dt.month

    df["is_weekend"] = (
        df["pickup_dayofweek"] >= 5
    ).astype(int)

    df["is_rush_hour"] = (
        (df["is_weekend"] == 0)
        & (
            df["pickup_hour"].isin(range(7, 10))
            | df["pickup_hour"].isin(range(17, 20))
        )
    ).astype(int)

    df["is_night"] = (
        df["pickup_hour"].isin(
            list(range(22, 24)) + list(range(0, 6))
        )
    ).astype(int)

    # -------------------------
    # Location features
    # -------------------------
    df["is_airport"] = (
        df["PULocationID"].isin(AIRPORT_ZONE_IDS)
        | df["DOLocationID"].isin(AIRPORT_ZONE_IDS)
    ).astype(int)

    #create index to add to feature store
    df = df.reset_index(drop=True).copy()
    df["trip_id"] = df.index

    logger.info(
        "Feature engineering completed: added %d new features.",
        len(df.columns) - n_cols_before,
    )

    return df


COLS_TO_DROP = [
    "total_amount",          # leakage
    "payment_type",          # constant after filtering
    "lpep_dropoff_datetime",
    "store_and_fwd_flag",
    "VendorID",
]

def upload_to_hopsworks(df: pd.DataFrame, parameters: Dict[str, Any]) -> None:
    to_feature_store = parameters["to_feature_store"]

    if to_feature_store:

        project = hopsworks.login(
            api_key_value=os.getenv("FS_API_KEY"),
            project=os.getenv("FS_PROJECT_NAME"),
        )

        fs = project.get_feature_store()

        fg = fs.get_or_create_feature_group(
            name="green_taxi_features",
            version=1,
            description="Green Taxi engineered features (2024-2025)",
            primary_key=["trip_id"],
            event_time="lpep_pickup_datetime",
            online_enabled=False,
            time_travel_format="HUDI",
        )

        fg.insert(
            features=df,
            overwrite=True, # type: ignore
            storage="offline", # type: ignore
            write_options={"wait_for_job": True},
        )

        fg.statistics_config = {
            "enabled": True,
            "histograms": True,
            "correlations": True,
        }

        fg.update_statistics_config()
        fg.compute_statistics()

        logger.info(
            "Uploaded %d trips to Hopsworks Feature Group '%s'.",
            len(df),
            fg.name,
        )
