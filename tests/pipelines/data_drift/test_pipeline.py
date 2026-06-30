import pandas as pd
import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from mlops_project.pipelines.data_drift import create_pipeline
from mlops_project.pipelines.data_drift.nodes import (
    REPORT_COLUMNS,
    _build_raw_monitoring_frames,
    detect_drift,
)
from mlops_project.pipelines.preprocessing_train.nodes import GreenTaxiPreprocessor

DATA_CLEANING_PARAMS = {
    "min_pickup_datetime": "2024-01-01",
    "min_trip_distance": 0,
    "max_trip_distance": 100,
    "min_fare_amount": 0,
    "max_fare_amount": 500,
    "min_duration_min": 1,
    "max_duration_min": 180,
    "min_tip_amount": 0,
    "max_tip_amount": 200,
    "min_location_id": 1,
    "max_location_id": 263,
}
DATA_DRIFT_PARAMS = {
    "timestamp_column": "lpep_pickup_datetime",
    "chunk_size": 5,
    "continuous_methods": ["jensen_shannon"],
    "categorical_methods": ["jensen_shannon"],
    "include_data_reconstruction": True,
}
WEEKEND_START_DAY = 5
DRIFT_ROW_COUNT = 20
PREPROCESSING_PARAMS = {
    "columns_to_drop": [
        "VendorID",
        "lpep_pickup_datetime",
        "lpep_dropoff_datetime",
        "store_and_fwd_flag",
        "payment_type",
        "total_amount",
        "source_partition",
        "source_year",
        "trip_id",
    ]
}


def _clean_feature_rows(
    start: str,
    rows: int,
    *,
    distance_offset: float = 0.0,
    borough: str = "Queens",
) -> pd.DataFrame:
    pickup = pd.date_range(start, periods=rows, freq="D")
    dropoff = pickup + pd.Timedelta(minutes=20)
    return pd.DataFrame(
        {
            "VendorID": [2] * rows,
            "lpep_pickup_datetime": pickup,
            "lpep_dropoff_datetime": dropoff,
            "store_and_fwd_flag": ["N"] * rows,
            "RatecodeID": [1] * rows,
            "PULocationID": [75] * rows,
            "DOLocationID": [74] * rows,
            "passenger_count": [1] * rows,
            "trip_distance": [1.0 + distance_offset + i * 0.1 for i in range(rows)],
            "fare_amount": [8.0 + distance_offset] * rows,
            "extra": [0.0] * rows,
            "mta_tax": [0.5] * rows,
            "tip_amount": [2.0] * rows,
            "tolls_amount": [0.0] * rows,
            "improvement_surcharge": [1.0] * rows,
            "total_amount": [11.5] * rows,
            "payment_type": [1] * rows,
            "trip_type": [1] * rows,
            "congestion_surcharge": [0.0] * rows,
            "PU_borough": [borough] * rows,
            "DO_borough": ["Manhattan"] * rows,
            "source_partition": [f"green_tripdata_{start[:4]}"] * rows,
            "source_year": [int(start[:4])] * rows,
            "trip_duration_min": [20.0] * rows,
            "pickup_hour": pickup.hour,
            "pickup_dayofweek": pickup.dayofweek,
            "pickup_month": pickup.month,
            "is_weekend": (pickup.dayofweek >= WEEKEND_START_DAY).astype(int),
            "is_rush_hour": [0] * rows,
            "is_night": [0] * rows,
            "is_airport": [0] * rows,
            "trip_id": list(range(rows)),
        }
    )


def _ingested_rows(
    start: str, rows: int, *, distance_offset: float = 0.0
) -> pd.DataFrame:
    clean = _clean_feature_rows(start, rows, distance_offset=distance_offset)
    return clean.drop(
        columns=[
            "trip_duration_min",
            "pickup_hour",
            "pickup_dayofweek",
            "pickup_month",
            "is_weekend",
            "is_rush_hour",
            "is_night",
            "is_airport",
            "trip_id",
        ]
    )


def _fitted_preprocessor() -> tuple[
    pd.DataFrame, pd.DataFrame, GreenTaxiPreprocessor, list[str]
]:
    reference_raw = _clean_feature_rows("2024-01-01", 20).drop(columns=["tip_amount"])
    transformer = GreenTaxiPreprocessor(PREPROCESSING_PARAMS, "tip_amount").fit(
        reference_raw
    )
    reference_preprocessed = transformer.transform(reference_raw).reset_index(drop=True)
    return (
        reference_raw,
        reference_preprocessed,
        transformer,
        reference_preprocessed.columns.tolist(),
    )


def test_data_drift_pipeline_uses_2026_drift_holdout_contract() -> None:
    pipeline = create_pipeline()
    node = pipeline.nodes[0]

    assert "drift_ingested_data" in node.inputs
    assert "X_batch_preprocessed" not in node.inputs
    assert set(node.outputs) == {
        "drift_report",
        "drift_summary",
        "X_drift_preprocessed",
    }


def test_build_raw_monitoring_frames_keeps_timestamp_and_fitted_schema() -> None:
    reference_raw, _, transformer, _ = _fitted_preprocessor()
    current_clean = _clean_feature_rows("2026-01-01", 10, distance_offset=5.0)

    reference, current, numeric_columns, categorical_columns = (
        _build_raw_monitoring_frames(
            reference_raw,
            current_clean,
            transformer,
            "lpep_pickup_datetime",
        )
    )

    assert "lpep_pickup_datetime" in reference.columns
    assert "trip_distance" in numeric_columns
    assert "PU_borough" in categorical_columns
    assert reference.columns.tolist() == current.columns.tolist()


def test_detect_drift_returns_report_summary_and_transformed_drift_batch() -> None:
    reference_raw, reference_preprocessed, transformer, production_columns = (
        _fitted_preprocessor()
    )
    drift_ingested = _ingested_rows("2026-01-01", DRIFT_ROW_COUNT, distance_offset=5.0)

    report, summary, drift_preprocessed = detect_drift(
        reference_raw,
        reference_preprocessed,
        drift_ingested,
        transformer,
        production_columns,
        DATA_CLEANING_PARAMS,
        "tip_amount",
        DATA_DRIFT_PARAMS,
    )

    assert report.columns.tolist() == REPORT_COLUMNS
    assert {"univariate", "data_reconstruction"}.issubset(set(report["drift_method"]))
    assert summary["current_rows"] == DRIFT_ROW_COUNT
    assert set(summary["alert_counts_by_method"]).issubset(
        {"univariate", "data_reconstruction"}
    )
    assert drift_preprocessed.columns.tolist() == production_columns
    assert len(drift_preprocessed) == DRIFT_ROW_COUNT


def test_detect_drift_fails_clearly_without_overlapping_raw_features() -> None:
    reference_raw, reference_preprocessed, transformer, production_columns = (
        _fitted_preprocessor()
    )
    drift_ingested = _ingested_rows("2026-01-01", 10)

    with pytest.raises(ValueError, match="No overlapping raw monitoring features"):
        detect_drift(
            reference_raw.drop(
                columns=transformer.numeric_features_
                + transformer.categorical_features_
            ),
            reference_preprocessed,
            drift_ingested,
            transformer,
            production_columns,
            DATA_CLEANING_PARAMS,
            "tip_amount",
            DATA_DRIFT_PARAMS,
        )


def test_data_drift_pipeline_creates_all_outputs() -> None:
    reference_raw, reference_preprocessed, transformer, production_columns = (
        _fitted_preprocessor()
    )
    catalog = DataCatalog(
        {
            "X_train_data": MemoryDataset(data=reference_raw),
            "X_train_preprocessed": MemoryDataset(data=reference_preprocessed),
            "drift_ingested_data": MemoryDataset(
                data=_ingested_rows("2026-01-01", DRIFT_ROW_COUNT, distance_offset=5.0)
            ),
            "preprocessing_transformer": MemoryDataset(data=transformer),
            "production_columns": MemoryDataset(data=production_columns),
            "params:data_cleaning": MemoryDataset(data=DATA_CLEANING_PARAMS),
            "params:target_column": MemoryDataset(data="tip_amount"),
            "params:data_drift": MemoryDataset(data=DATA_DRIFT_PARAMS),
            "drift_report": MemoryDataset(),
            "drift_summary": MemoryDataset(),
            "X_drift_preprocessed": MemoryDataset(),
        }
    )

    SequentialRunner().run(create_pipeline(), catalog)

    assert catalog.load("drift_report").columns.tolist() == REPORT_COLUMNS
    assert catalog.load("drift_summary")["current_rows"] == DRIFT_ROW_COUNT
    assert catalog.load("X_drift_preprocessed").columns.tolist() == production_columns
