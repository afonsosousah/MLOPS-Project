import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

AIRPORT_ZONE_IDS = {1, 132, 138}

COLS_TO_DROP = [
    "tip_amount",
    "total_amount",
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "payment_type",
    "store_and_fwd_flag",
    "VendorID",
]


# ---------------------------------------------------------------------------
# Step 1 — Clean
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to credit-card trips and remove invalid records.

    Keeps only payment_type == 1 (credit card) since cash tips are not recorded.
    Removes trips with invalid distances, fares, timestamps, or extreme durations.
    """
    n = len(df)
    df = df[df["payment_type"] == 1].copy()
    logger.info("After credit card filter: %d rows (%.1f%%)", len(df), len(df) / n * 100)

    df = df.dropna(subset=["lpep_pickup_datetime", "lpep_dropoff_datetime",
                            "PULocationID", "DOLocationID", "trip_distance", "fare_amount"])

    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 500)]
    df = df[pd.to_datetime(df["lpep_pickup_datetime"]) >= "2024-01-01"]

    pickup = pd.to_datetime(df["lpep_pickup_datetime"])
    dropoff = pd.to_datetime(df["lpep_dropoff_datetime"])
    duration_min = (dropoff - pickup).dt.total_seconds() / 60
    df = df[(duration_min >= 1) & (duration_min <= 180)]

    df = df.reset_index(drop=True)
    logger.info("After cleaning: %d rows remaining.", len(df))
    return df


# ---------------------------------------------------------------------------
# Step 2 — Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive temporal, efficiency, and location features from raw trip columns.
    Expects PU_borough / DO_borough already joined by ingestion pipeline.
    """
    df = df.copy()

    pickup = pd.to_datetime(df["lpep_pickup_datetime"])
    dropoff = pd.to_datetime(df["lpep_dropoff_datetime"])

    df["trip_duration_min"] = (dropoff - pickup).dt.total_seconds() / 60
    df["pickup_hour"]       = pickup.dt.hour
    df["pickup_dayofweek"]  = pickup.dt.dayofweek
    df["pickup_month"]      = pickup.dt.month

    df["is_weekend"]   = (df["pickup_dayofweek"] >= 5).astype(int)
    df["is_rush_hour"] = (
        (df["is_weekend"] == 0) &
        (df["pickup_hour"].isin(range(7, 10)) | df["pickup_hour"].isin(range(17, 20)))
    ).astype(int)
    df["is_night"] = df["pickup_hour"].isin(list(range(22, 24)) + list(range(0, 6))).astype(int)

    df["speed_mph"]     = (df["trip_distance"] / (df["trip_duration_min"] / 60)).clip(0, 80)
    df["fare_per_mile"] = (df["fare_amount"] / df["trip_distance"]).clip(0, 50)

    df["is_airport"]   = (
        df["PULocationID"].isin(AIRPORT_ZONE_IDS) | df["DOLocationID"].isin(AIRPORT_ZONE_IDS)
    ).astype(int)
    df["same_borough"] = (df["PU_borough"] == df["DO_borough"]).astype(int)

    logger.info("Feature engineering done. %d columns total.", len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Step 3 — Encode + target + drop leakage
# ---------------------------------------------------------------------------

def create_target_and_encode(df: pd.DataFrame) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """
    Create is_tipped target, one-hot encode borough columns, drop leakage columns.

    Returns:
        df_final: Model-ready DataFrame with target column.
        encoder: Fitted OneHotEncoder (saved as encoder_transform in catalog).
    """
    df = df.copy()

    # Target
    df["is_tipped"] = (df["tip_amount"] > 0).astype(int)

    # Encode borough columns
    cat_cols = ["PU_borough", "DO_borough"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cat_cols),
        index=df.index,
    )

    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded_df], axis=1)

    # Drop leakage and raw columns
    cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    df = df.fillna(-1)
    logger.info("Preprocessed training data: %d rows, %d features + target.", len(df), len(df.columns) - 1)
    return df, encoder


# ---------------------------------------------------------------------------
# Kedro node entry point
# ---------------------------------------------------------------------------

def preprocess_train(ref_data: pd.DataFrame) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """
    Full preprocessing for training data:
    clean → feature engineer → encode + target.

    Args:
        ref_data: 2024 reference data from split_data pipeline.

    Returns:
        preprocessed_training_data: Model-ready DataFrame.
        encoder_transform: Fitted OneHotEncoder for reuse in preprocessing_batch.
    """
    df = clean_data(ref_data)
    df = engineer_features(df)
    df, encoder = create_target_and_encode(df)
    return df, encoder
