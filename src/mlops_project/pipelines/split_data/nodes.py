import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def split_data(
    df: pd.DataFrame,
    split_date: str,
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split filtered Green Taxi data into chronological train/test sets."""
    data = df.copy()
    data["lpep_pickup_datetime"] = pd.to_datetime(data["lpep_pickup_datetime"])
    split_timestamp = pd.Timestamp(split_date)

    train_df = data[data["lpep_pickup_datetime"] < split_timestamp].copy()
    test_df = data[data["lpep_pickup_datetime"] >= split_timestamp].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Split at '{split_date}' produced an empty train or test set. "
            "check the date against the actual data range."
        )

    logger.info(
        "Split at %s -> train: %d rows (%s -> %s) | test: %d rows (%s -> %s)",
        split_date,
        len(train_df),
        train_df["lpep_pickup_datetime"].min(),
        train_df["lpep_pickup_datetime"].max(),
        len(test_df),
        test_df["lpep_pickup_datetime"].min(),
        test_df["lpep_pickup_datetime"].max(),
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
