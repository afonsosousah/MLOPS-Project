import pandas as pd
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from mlops_project.pipelines.data_drift import create_pipeline
from mlops_project.pipelines.data_drift.nodes import REPORT_COLUMNS, detect_drift

DATA_DRIFT_PARAMS = {"p_value_threshold": 1.0}


def _reference_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_distance": [1.0, 1.1, 1.2, 1.3],
            "PU_borough": ["Queens", "Queens", "Brooklyn", "Brooklyn"],
        }
    )


def _current_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_distance": [10.0, 10.1, 10.2, 10.3],
            "PU_borough": ["Queens", "Queens", "Queens", "Queens"],
        }
    )


def test_detect_drift_returns_stable_report_schema() -> None:
    report = detect_drift(_reference_data(), _current_data(), DATA_DRIFT_PARAMS)

    assert report.columns.tolist() == REPORT_COLUMNS
    assert set(report["test"]) == {"ks", "chi2"}
    assert report["drift_detected"].all()


def test_detect_drift_returns_empty_report_with_schema_when_no_columns_overlap() -> (
    None
):
    report = detect_drift(
        pd.DataFrame({"reference_only": [1, 2]}),
        pd.DataFrame({"current_only": [1, 2]}),
        DATA_DRIFT_PARAMS,
    )

    assert report.columns.tolist() == REPORT_COLUMNS
    assert report.empty


def test_data_drift_pipeline_creates_report() -> None:
    catalog = DataCatalog(
        {
            "X_train_preprocessed": MemoryDataset(data=_reference_data()),
            "X_batch_preprocessed": MemoryDataset(data=_current_data()),
            "params:data_drift": MemoryDataset(data=DATA_DRIFT_PARAMS),
            "drift_report": MemoryDataset(),
        }
    )

    SequentialRunner().run(create_pipeline(), catalog)

    report = catalog.load("drift_report")
    assert report.columns.tolist() == REPORT_COLUMNS
