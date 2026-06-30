import importlib
import logging
import os
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

REPORT_COLUMNS = ["status", "rows", "feature_group", "version", "message"]


def _load_dotenv_if_available() -> None:
    try:
        dotenv = importlib.import_module("dotenv")
    except ModuleNotFoundError:
        return

    dotenv.load_dotenv()


def _build_report(
    status: str,
    rows: int,
    feature_group: str,
    version: int,
    message: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "status": status,
                "rows": rows,
                "feature_group": feature_group,
                "version": version,
                "message": message,
            }
        ],
        columns=REPORT_COLUMNS,
    )


def _feature_store_credentials(parameters: dict[str, Any]) -> tuple[str, str]:
    credential_env_vars = parameters.get("credential_env_vars", {})
    api_key_var = credential_env_vars.get("api_key", "FS_API_KEY")
    project_name_var = credential_env_vars.get("project_name", "FS_PROJECT_NAME")

    api_key = os.getenv(api_key_var, "")
    project_name = os.getenv(project_name_var, "")

    if not api_key or not project_name:
        missing = [
            var_name
            for var_name, value in (
                (api_key_var, api_key),
                (project_name_var, project_name),
            )
            if not value
        ]
        raise ValueError(
            "Missing Hopsworks credentials in environment variables: "
            + ", ".join(missing)
        )

    return api_key, project_name


def upload_features_to_store(
    features: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Optionally upload engineered Green Taxi features to Hopsworks."""
    feature_group_name = parameters.get("feature_group_name", "green_taxi_features")
    feature_group_version = int(parameters.get("feature_group_version", 1))

    if not parameters.get("enabled", False):
        message = "Feature-store upload skipped because feature_store.enabled is false."
        logger.info(message)
        return _build_report(
            status="skipped",
            rows=len(features),
            feature_group=feature_group_name,
            version=feature_group_version,
            message=message,
        )

    _load_dotenv_if_available()
    api_key, project_name = _feature_store_credentials(parameters)

    hopsworks = importlib.import_module("hopsworks")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    feature_store = project.get_feature_store()
    feature_group = feature_store.get_or_create_feature_group(
        name=feature_group_name,
        version=feature_group_version,
        description=parameters.get("description", "Green Taxi engineered features"),
        primary_key=parameters.get("primary_key", ["trip_id"]),
        event_time=parameters.get("event_time", "lpep_pickup_datetime"),
        online_enabled=parameters.get("online_enabled", False),
        time_travel_format=parameters.get("time_travel_format", "HUDI"),
    )

    insert_kwargs = {
        "features": features,
        "overwrite": parameters.get("overwrite", True),
        "storage": parameters.get("storage", "offline"),
        "write_options": {"wait_for_job": parameters.get("wait_for_job", True)},
    }
    feature_group.insert(**insert_kwargs)

    if parameters.get("compute_statistics", True):
        feature_group.statistics_config = parameters.get(
            "statistics_config",
            {"enabled": True, "histograms": True, "correlations": True},
        )
        try:
            feature_group.update_statistics_config()
            feature_group.compute_statistics()
        except Exception as exc:
            logger.warning(
                "Could not compute feature group statistics (non-fatal): %s", exc
            )

    message = f"Uploaded {len(features)} rows to Hopsworks feature group."
    logger.info(
        "Uploaded %d trips to Hopsworks Feature Group '%s'.",
        len(features),
        feature_group.name or feature_group_name,
    )
    return _build_report(
        status="uploaded",
        rows=len(features),
        feature_group=feature_group.name or feature_group_name,
        version=feature_group_version,
        message=message,
    )
