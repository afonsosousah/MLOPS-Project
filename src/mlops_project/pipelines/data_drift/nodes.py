import logging
from typing import Any

import nannyml as nml
import pandas as pd

from mlops_project.pipelines.data_cleaning.nodes import data_cleaning

logger = logging.getLogger(__name__)

REPORT_COLUMNS = [
    "drift_method",
    "column",
    "method",
    "chunk_key",
    "chunk_index",
    "start_date",
    "end_date",
    "period",
    "value",
    "upper_threshold",
    "lower_threshold",
    "alert",
]
THREE_LEVEL_METRIC_KEY_LENGTH = 3


def _to_python(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _stringify_date(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


def _chunk_value(row: pd.Series, column: str) -> Any:
    for key in (("chunk", "chunk", column), ("chunk", column)):
        if key in row.index:
            return row[key]
    return None


def _metric_value(row: pd.Series, keys: tuple[str, ...]) -> Any:
    if keys in row.index:
        return row[keys]
    if len(keys) == THREE_LEVEL_METRIC_KEY_LENGTH:
        two_level_key = (keys[0], keys[2])
        if two_level_key in row.index:
            return row[two_level_key]
    return None


def _nannyml_results_to_report(
    results_df: pd.DataFrame,
    drift_method: str,
) -> pd.DataFrame:
    """Flatten NannyML's MultiIndex result frame into one row per metric/chunk."""
    rows: list[dict[str, Any]] = []

    metric_keys = [
        column
        for column in results_df.columns
        if isinstance(column, tuple) and column[0] != "chunk" and column[-1] == "value"
    ]

    for _, row in results_df.iterrows():
        for metric_key in metric_keys:
            if len(metric_key) == THREE_LEVEL_METRIC_KEY_LENGTH:
                column_name, method, _ = metric_key
            else:
                column_name, _ = metric_key
                method = column_name

            rows.append(
                {
                    "drift_method": drift_method,
                    "column": str(column_name),
                    "method": str(method),
                    "chunk_key": _to_python(_chunk_value(row, "key")),
                    "chunk_index": _to_python(_chunk_value(row, "chunk_index")),
                    "start_date": _stringify_date(_chunk_value(row, "start_date")),
                    "end_date": _stringify_date(_chunk_value(row, "end_date")),
                    "period": _to_python(_chunk_value(row, "period")),
                    "value": _to_python(
                        _metric_value(row, (*metric_key[:-1], "value"))
                    ),
                    "upper_threshold": _to_python(
                        _metric_value(row, (*metric_key[:-1], "upper_threshold"))
                    ),
                    "lower_threshold": _to_python(
                        _metric_value(row, (*metric_key[:-1], "lower_threshold"))
                    ),
                    "alert": bool(_metric_value(row, (*metric_key[:-1], "alert"))),
                }
            )

    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def _get_monitoring_columns(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    columns: list[str],
) -> list[str]:
    return [
        column
        for column in columns
        if column in reference_data.columns and column in current_data.columns
    ]


def _build_raw_monitoring_frames(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    preprocessing_transformer: Any,
    timestamp_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    numeric_columns = _get_monitoring_columns(
        reference_data,
        current_data,
        getattr(preprocessing_transformer, "numeric_features_", []),
    )
    categorical_columns = _get_monitoring_columns(
        reference_data,
        current_data,
        getattr(preprocessing_transformer, "categorical_features_", []),
    )
    feature_columns = numeric_columns + categorical_columns

    if not feature_columns:
        raise ValueError("No overlapping raw monitoring features were found.")
    if timestamp_column not in reference_data.columns:
        raise ValueError(
            f"Reference data is missing timestamp column '{timestamp_column}'."
        )
    if timestamp_column not in current_data.columns:
        raise ValueError(
            f"Current data is missing timestamp column '{timestamp_column}'."
        )

    reference_monitoring = reference_data[[timestamp_column, *feature_columns]].copy()
    current_monitoring = current_data[[timestamp_column, *feature_columns]].copy()
    reference_monitoring[timestamp_column] = pd.to_datetime(
        reference_monitoring[timestamp_column]
    )
    current_monitoring[timestamp_column] = pd.to_datetime(
        current_monitoring[timestamp_column]
    )

    return (
        reference_monitoring,
        current_monitoring,
        numeric_columns,
        categorical_columns,
    )


def _build_encoded_monitoring_frames(  # noqa: PLR0913
    reference_encoded: pd.DataFrame,
    current_encoded: pd.DataFrame,
    reference_timestamps: pd.Series,
    current_timestamps: pd.Series,
    production_columns: list[str],
    timestamp_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    encoded_columns = [
        column
        for column in production_columns
        if column in reference_encoded.columns and column in current_encoded.columns
    ]
    if not encoded_columns:
        raise ValueError("No overlapping encoded monitoring features were found.")

    reference_monitoring = reference_encoded.loc[:, encoded_columns].copy()
    current_monitoring = current_encoded.loc[:, encoded_columns].copy()
    reference_monitoring[timestamp_column] = pd.to_datetime(
        reference_timestamps.reset_index(drop=True)
    )
    current_monitoring[timestamp_column] = pd.to_datetime(
        current_timestamps.reset_index(drop=True)
    )
    return reference_monitoring, current_monitoring, encoded_columns


def _run_univariate_drift(  # noqa: PLR0913
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    timestamp_column: str,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    feature_columns = numeric_columns + categorical_columns
    calculator = nml.UnivariateDriftCalculator(
        column_names=feature_columns,
        treat_as_categorical=categorical_columns or None,
        timestamp_column_name=timestamp_column,
        continuous_methods=parameters.get("continuous_methods", ["jensen_shannon"]),
        categorical_methods=parameters.get("categorical_methods", ["jensen_shannon"]),
        chunk_size=parameters.get("chunk_size"),
        chunk_number=parameters.get("chunk_number"),
        chunk_period=parameters.get("chunk_period", "M"),
    )
    calculator.fit(reference_data)
    results = calculator.calculate(current_data)
    return _nannyml_results_to_report(
        results.filter(period="analysis").to_df(),
        drift_method="univariate",
    )


def _run_data_reconstruction_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    encoded_columns: list[str],
    timestamp_column: str,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    calculator = nml.DataReconstructionDriftCalculator(
        column_names=encoded_columns,
        timestamp_column_name=timestamp_column,
        chunk_size=parameters.get("chunk_size"),
        chunk_number=parameters.get("chunk_number"),
        chunk_period=parameters.get("chunk_period", "M"),
    )
    calculator.fit(reference_data)
    results = calculator.calculate(current_data)
    return _nannyml_results_to_report(
        results.filter(period="analysis").to_df(),
        drift_method="data_reconstruction",
    )


def _summarize_drift(  # noqa: PLR0913
    report: pd.DataFrame,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    encoded_columns: list[str],
    timestamp_column: str,
) -> dict[str, Any]:
    alert_counts = (
        report.groupby("drift_method")["alert"].sum().astype(int).to_dict()
        if not report.empty
        else {}
    )
    chunk_counts = (
        report.groupby("drift_method")["chunk_index"].nunique().astype(int).to_dict()
        if not report.empty
        else {}
    )
    return {
        "reference_rows": int(len(reference_data)),
        "current_rows": int(len(current_data)),
        "reference_start": _stringify_date(reference_data[timestamp_column].min()),
        "reference_end": _stringify_date(reference_data[timestamp_column].max()),
        "current_start": _stringify_date(current_data[timestamp_column].min()),
        "current_end": _stringify_date(current_data[timestamp_column].max()),
        "numeric_feature_count": int(len(numeric_columns)),
        "categorical_feature_count": int(len(categorical_columns)),
        "encoded_feature_count": int(len(encoded_columns)),
        "alert_counts_by_method": alert_counts,
        "chunk_counts_by_method": chunk_counts
    }


def detect_drift(  # noqa: PLR0913
    reference_raw_data: pd.DataFrame,
    reference_preprocessed_data: pd.DataFrame,
    drift_ingested_data: pd.DataFrame,
    preprocessing_transformer: Any,
    production_columns: list[str],
    data_cleaning_parameters: dict[str, Any],
    target_column: str,
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Run NannyML drift monitoring against the 2026 Green Taxi holdout."""
    timestamp_column = parameters.get("timestamp_column", "lpep_pickup_datetime")
    current_clean = data_cleaning(drift_ingested_data, data_cleaning_parameters)
    current_preprocessed = preprocessing_transformer.transform(current_clean)
    current_preprocessed = current_preprocessed.loc[:, list(production_columns)]
    current_preprocessed = current_preprocessed.reset_index(drop=True)

    raw_reference, raw_current, numeric_columns, categorical_columns = (
        _build_raw_monitoring_frames(
            reference_raw_data,
            current_clean,
            preprocessing_transformer,
            timestamp_column,
        )
    )
    encoded_reference, encoded_current, encoded_columns = (
        _build_encoded_monitoring_frames(
            reference_preprocessed_data.reset_index(drop=True),
            current_preprocessed,
            raw_reference[timestamp_column],
            raw_current[timestamp_column],
            list(production_columns),
            timestamp_column,
        )
    )

    report_parts = [
        _run_univariate_drift(
            raw_reference,
            raw_current,
            numeric_columns,
            categorical_columns,
            timestamp_column,
            parameters,
        )
    ]

    if parameters.get("include_data_reconstruction", True):
        report_parts.append(
            _run_data_reconstruction_drift(
                encoded_reference,
                encoded_current,
                encoded_columns,
                timestamp_column,
                parameters,
            )
        )

    report = pd.concat(report_parts, ignore_index=True)
    summary = _summarize_drift(
        report,
        raw_reference,
        raw_current,
        numeric_columns,
        categorical_columns,
        encoded_columns,
        timestamp_column,
    )
    logger.info(
        "NannyML drift monitoring complete: %d report rows, alert counts %s.",
        len(report),
        summary["alert_counts_by_method"],
    )
    return report, summary, current_preprocessed
