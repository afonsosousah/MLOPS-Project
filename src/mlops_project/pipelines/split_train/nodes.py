import logging

import pandas as pd

logger = logging.getLogger(__name__)


def split_train(
    train_data: pd.DataFrame,
    split_date: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split training-period data into chronological train/validation X/y sets."""
    data = train_data.copy()
    data["lpep_pickup_datetime"] = pd.to_datetime(data["lpep_pickup_datetime"])

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' is missing from train data.")

    split_timestamp = pd.Timestamp(split_date)
    train_mask = data["lpep_pickup_datetime"] < split_timestamp
    val_mask = data["lpep_pickup_datetime"] >= split_timestamp

    train_df = data[train_mask].copy()
    val_df = data[val_mask].copy()

    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Train/validation split at '{split_date}' produced an empty split."
        )

    X_train = train_df.drop(columns=[target_column]).reset_index(drop=True)
    X_val = val_df.drop(columns=[target_column]).reset_index(drop=True)
    y_train = train_df[[target_column]].reset_index(drop=True)
    y_val = val_df[[target_column]].reset_index(drop=True)

    logger.info(
        "Train/validation split at %s -> train: %d rows | val: %d rows.",
        split_date,
        len(X_train),
        len(X_val),
    )
    return X_train, X_val, y_train, y_val
