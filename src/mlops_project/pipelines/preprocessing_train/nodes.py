import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

# Airport zone IDs: EWR=1, JFK=132, LGA=138
AIRPORT_ZONE_IDS = {1, 132, 138}

# Columns to drop before training (leakage + raw temporals)
COLS_TO_DROP = [
    "tip_amount",             # used to create target — leakage
    "total_amount",           # includes tip — leakage
    "lpep_pickup_datetime",   # replaced by engineered features
    "lpep_dropoff_datetime",  # replaced by engineered features
    "payment_type",           # constant after filtering (all == 1)
    "store_and_fwd_flag",     # administrative
    "VendorID",               # administrative
    "source_month",           # metadata column added during loading
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to credit-card trips and remove invalid/outlier rows.

    - Drops ehail_fee if still present (already dropped in ingestion, safe to repeat).
    - Filters to payment_type == 1 (credit card only; cash tips are not recorded).
    - Drops rows missing essential fields.
    - Removes physically impossible or extreme trips.
    """
    df = df.drop(columns=["ehail_fee"], errors="ignore")

    n_start = len(df)
    df = df[df["payment_type"] == 1].copy()
    logger.info("After credit-card filter: %d rows (%.1f%%)", len(df), len(df) / n_start * 100)

    df = df.dropna(
        subset=["lpep_pickup_datetime", "lpep_dropoff_datetime",
                "PULocationID", "DOLocationID", "trip_distance", "fare_amount"]
    )

    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 500)]
    df = df[df["lpep_pickup_datetime"] >= "2024-01-01"]
    df = df[df["lpep_dropoff_datetime"] > df["lpep_pickup_datetime"]]

    duration_min = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(duration_min >= 1) & (duration_min <= 180)]

    logger.info("After cleaning: %d rows remaining", len(df))
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal and trip-efficiency features.

    Borough columns (PU_borough, DO_borough) are already present from the
    ingestion pipeline, so no zone-lookup join is needed here.
    """
    df = df.copy()
    pickup = pd.to_datetime(df["lpep_pickup_datetime"])
    dropoff = pd.to_datetime(df["lpep_dropoff_datetime"])

    df["trip_duration_min"] = (dropoff - pickup).dt.total_seconds() / 60
    df["pickup_hour"] = pickup.dt.hour
    df["pickup_dayofweek"] = pickup.dt.dayofweek   # 0=Monday, 6=Sunday
    df["pickup_month"] = pickup.dt.month

    df["is_weekend"] = (df["pickup_dayofweek"] >= 5).astype(int)
    df["is_rush_hour"] = (
        (df["is_weekend"] == 0) &
        (df["pickup_hour"].isin(range(7, 10)) | df["pickup_hour"].isin(range(17, 20)))
    ).astype(int)
    df["is_night"] = (
        df["pickup_hour"].isin(list(range(22, 24)) + list(range(0, 6)))
    ).astype(int)

    df["speed_mph"] = (df["trip_distance"] / (df["trip_duration_min"] / 60)).clip(0, 80)
    df["fare_per_mile"] = (df["fare_amount"] / df["trip_distance"]).clip(0, 50)

    df["is_airport"] = (
        df["PULocationID"].isin(AIRPORT_ZONE_IDS) | df["DOLocationID"].isin(AIRPORT_ZONE_IDS)
    ).astype(int)
    df["same_borough"] = (df["PU_borough"] == df["DO_borough"]).astype(int)

    return df


def preprocess_train(
    ref_data: pd.DataFrame,
    parameters: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    Full training preprocessing: clean → target → features → encode → split.

    Returns X_train, X_test, y_train, y_test (as single-column DataFrames), encoder.
    y_train/y_test are saved as DataFrames so Kedro's CSVDataset can round-trip them;
    the modeling node can call .squeeze() or ["is_tipped"] to get a Series.
    """
    target = parameters["target_column"]
    test_size = parameters.get("test_size", 0.2)
    random_state = parameters.get("random_state", 42)

    df = clean_data(ref_data)
    df["is_tipped"] = (df["tip_amount"] > 0).astype(int)
    df = engineer_features(df)

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

    cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    df = df.fillna(-1)

    X = df.drop(columns=[target])
    y = df[[target]]   # keep as DataFrame for catalog compatibility

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        "Train: %d rows | Test: %d rows | Features: %d",
        len(X_train), len(X_test), X_train.shape[1],
    )
    return X_train, X_test, y_train, y_test, encoder
