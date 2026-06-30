import logging
from typing import Any
import importlib
import os

import pandas as pd

logger = logging.getLogger(__name__)



def _load_dotenv_if_available() -> None:
    try:
        dotenv = importlib.import_module("dotenv")
    except ModuleNotFoundError:
        return
    dotenv.load_dotenv()


def read_features_from_store(parameters: dict[str, Any]) -> pd.DataFrame:
    """Read the engineered Green Taxi features back from the Hopsworks feature store."""
    _load_dotenv_if_available()

    api_key = os.getenv(
        parameters.get("credential_env_vars", {}).get("api_key", "FS_API_KEY")
    )
    project_name = os.getenv(
        parameters.get("credential_env_vars", {}).get(
            "project_name", "FS_PROJECT_NAME"
        )
    )
    if not api_key or not project_name:
        raise ValueError("Missing Hopsworks credentials in environment variables.")

    hopsworks = importlib.import_module("hopsworks")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    feature_store = project.get_feature_store()

    feature_group_name = parameters.get("feature_group_name", "green_taxi_features")
    feature_group_version = int(parameters.get("feature_group_version", 1))

    feature_group = feature_store.get_feature_group(
        name=feature_group_name, version=feature_group_version
    )
    data = feature_group.read()

    logger.info(
        "Read %d rows from Hopsworks feature group '%s' (v%d).",
        len(data),
        feature_group_name,
        feature_group_version,
    )
    return data



def get_features(
    parameters: dict[str, Any],
    local_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Get engineered features either from Hopsworks or from a local dataset."""
    use_feature_store = parameters.get("use_feature_store", False)

    if not use_feature_store:
        if local_features is None:
            raise ValueError(
                "use_feature_store is false but no local_features dataset was provided."
            )
        logger.info("Using local features dataset (%d rows).", len(local_features))
        return local_features

    return read_features_from_store(parameters)







def split_data(
    df: pd.DataFrame,
    split_date: str,
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
