import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def split_data(df: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split feature data into train and validation sets by pickup date."""
    df = df.copy()
    df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])

    train_df = df[df["lpep_pickup_datetime"] < split_date].copy()
    val_df = df[df["lpep_pickup_datetime"] >= split_date].copy()

    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Split at '{split_date}' produced an empty train or validation set — "
            "check the date against the actual data range."
        )

    logger.info(
        "Split at %s -> train: %d rows (%s -> %s) | val: %d rows (%s -> %s)",
        split_date,
        len(train_df), train_df["lpep_pickup_datetime"].min(), train_df["lpep_pickup_datetime"].max(),
        len(val_df), val_df["lpep_pickup_datetime"].min(), val_df["lpep_pickup_datetime"].max(),
    )

    return train_df, val_df
