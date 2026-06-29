import pandas as pd
import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from mlops_project.pipelines.feature_store import create_pipeline
from mlops_project.pipelines.feature_store.nodes import upload_features_to_store

FEATURE_STORE_PARAMS = {
    "enabled": False,
    "feature_group_name": "green_taxi_features",
    "feature_group_version": 1,
}


def _engineered_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_id": [0, 1],
            "lpep_pickup_datetime": pd.to_datetime(
                ["2024-01-01 08:00", "2024-01-01 09:00"]
            ),
            "trip_distance": [1.5, 2.0],
        }
    )


def test_feature_store_upload_skips_when_disabled() -> None:
    report = upload_features_to_store(_engineered_data(), FEATURE_STORE_PARAMS)

    assert report["status"].tolist() == ["skipped"]
    assert report["rows"].tolist() == [2]
    assert report["feature_group"].tolist() == ["green_taxi_features"]


def test_feature_store_pipeline_writes_skip_report_when_disabled() -> None:
    catalog = DataCatalog(
        {
            "X_train_preprocessed": MemoryDataset(data=_engineered_data()),
            "params:feature_store": MemoryDataset(data=FEATURE_STORE_PARAMS),
            "feature_store_upload_report": MemoryDataset(),
        }
    )

    SequentialRunner().run(create_pipeline(), catalog)

    report = catalog.load("feature_store_upload_report")
    assert report["status"].tolist() == ["skipped"]


def test_feature_store_enabled_requires_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FS_API_KEY", raising=False)
    monkeypatch.delenv("FS_PROJECT_NAME", raising=False)

    params = {
        **FEATURE_STORE_PARAMS,
        "enabled": True,
        "credential_env_vars": {
            "api_key": "FS_API_KEY",
            "project_name": "FS_PROJECT_NAME",
        },
    }

    with pytest.raises(ValueError, match="Missing Hopsworks credentials"):
        upload_features_to_store(_engineered_data(), params)
