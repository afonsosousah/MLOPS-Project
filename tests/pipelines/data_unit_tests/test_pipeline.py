import pandas as pd
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from mlops_project.pipelines.data_unit_tests import create_pipeline
from mlops_project.pipelines.data_unit_tests.nodes import unit_test

EXPECTED_REPORT_COLUMNS = [
    "success",
    "severity",
    "expectation_type",
    "column",
    "min_value",
    "max_value",
    "value_set",
    "unexpected_count",
    "unexpected_percent",
    "observed_value",
]


def _valid_green_taxi_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "VendorID": [2, 2, 2],
            "lpep_pickup_datetime": pd.to_datetime(
                ["2024-01-01 08:00", "2024-01-01 09:00", "2024-01-01 10:00"]
            ),
            "lpep_dropoff_datetime": pd.to_datetime(
                ["2024-01-01 08:15", "2024-01-01 09:20", "2024-01-01 10:30"]
            ),
            "store_and_fwd_flag": ["N", "N", "Y"],
            "RatecodeID": [1, 1, 2],
            "PULocationID": [7, 41, 75],
            "DOLocationID": [42, 74, 236],
            "passenger_count": [1, 2, 1],
            "trip_distance": [1.2, 2.5, 3.0],
            "fare_amount": [8.0, 12.5, 16.0],
            "extra": [1.0, 1.0, 1.0],
            "mta_tax": [0.5, 0.5, 0.5],
            "tip_amount": [2.0, 3.0, 4.0],
            "tolls_amount": [0.0, 0.0, 0.0],
            "improvement_surcharge": [1.0, 1.0, 1.0],
            "total_amount": [12.5, 17.0, 21.5],
            "payment_type": [1, 1, 1],
            "trip_type": [1, 1, 1],
            "congestion_surcharge": [2.75, 2.75, 2.75],
        }
    )


def test_unit_test_returns_pass_report_for_valid_data() -> None:
    report = unit_test(_valid_green_taxi_data())

    assert report.columns.tolist() == EXPECTED_REPORT_COLUMNS
    assert not report.empty
    assert report["severity"].eq("pass").all()
    assert report["success"].all()


def test_unit_test_reports_warnings_without_raising() -> None:
    invalid_data = _valid_green_taxi_data()
    invalid_data.loc[0, "payment_type"] = 9
    invalid_data.loc[1, "fare_amount"] = -1
    invalid_data.loc[2, "lpep_dropoff_datetime"] = pd.Timestamp("2024-01-01 09:00")

    report = unit_test(invalid_data)

    assert report.columns.tolist() == EXPECTED_REPORT_COLUMNS
    assert (report["severity"] == "warning").any()
    assert (~report["success"]).any()


def test_data_unit_tests_pipeline_creates_modeling_and_drift_reports() -> None:
    catalog = DataCatalog(
        {
            "ingested_data": MemoryDataset(data=_valid_green_taxi_data()),
            "drift_ingested_data": MemoryDataset(data=_valid_green_taxi_data()),
            "reporting_tests": MemoryDataset(),
            "drift_reporting_tests": MemoryDataset(),
        }
    )

    SequentialRunner().run(create_pipeline(), catalog)

    modeling_report = catalog.load("reporting_tests")
    drift_report = catalog.load("drift_reporting_tests")

    assert modeling_report.columns.tolist() == EXPECTED_REPORT_COLUMNS
    assert drift_report.columns.tolist() == EXPECTED_REPORT_COLUMNS
    assert not modeling_report.empty
    assert not drift_report.empty
    assert modeling_report["severity"].eq("pass").all()
    assert drift_report["severity"].eq("pass").all()
