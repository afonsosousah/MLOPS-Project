"""Profiling helpers for NYC TLC Green Taxi parquet files."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import pandas as pd

GREEN_TAXI_PATTERN = re.compile(r"green_tripdata_(?P<year>\d{4})-(?P<month>\d{2})\.parquet$")


def parse_green_taxi_month(path: str | Path) -> dict[str, int | str]:
    """Extract year, month, and YYYY-MM period from a Green Taxi parquet path."""
    file_name = Path(path).name
    match = GREEN_TAXI_PATTERN.fullmatch(file_name)
    if not match:
        raise ValueError(f"Not a Green Taxi monthly parquet file: {file_name}")

    year = int(match.group("year"))
    month = int(match.group("month"))
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month in Green Taxi file name: {file_name}")

    return {"year": year, "month": month, "period": f"{year:04d}-{month:02d}"}


def discover_green_taxi_files(raw_dir: str | Path) -> pd.DataFrame:
    """Return discovered monthly parquet files with parsed period metadata."""
    raw_path = Path(raw_dir)
    rows: list[dict[str, object]] = []
    for path in sorted(raw_path.glob("*/*.parquet")):
        parsed = parse_green_taxi_month(path)
        rows.append(
            {
                **parsed,
                "path": str(path),
                "file_size_bytes": path.stat().st_size,
            }
        )

    files = pd.DataFrame(rows)
    if files.empty:
        return pd.DataFrame(columns=["year", "month", "period", "path", "file_size_bytes"])
    return files.sort_values(["year", "month"]).reset_index(drop=True)


def read_parquet_metadata(files: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read row counts and schema metadata without loading full parquet contents."""
    import pyarrow.parquet as pq

    monthly_rows: list[dict[str, object]] = []
    schema_rows: list[dict[str, object]] = []

    for record in files.to_dict("records"):
        parquet_file = pq.ParquetFile(record["path"])
        columns = []
        for field in parquet_file.schema_arrow:
            columns.append(field.name)
            schema_rows.append(
                {
                    "period": record["period"],
                    "year": record["year"],
                    "month": record["month"],
                    "column": field.name,
                    "dtype": str(field.type),
                }
            )

        monthly_rows.append(
            {
                "period": record["period"],
                "year": record["year"],
                "month": record["month"],
                "path": record["path"],
                "file_size_bytes": record["file_size_bytes"],
                "row_count": parquet_file.metadata.num_rows,
                "column_count": len(columns),
                "columns": ", ".join(columns),
            }
        )

    return pd.DataFrame(monthly_rows), pd.DataFrame(schema_rows)


def schema_presence_summary(schema: pd.DataFrame) -> pd.DataFrame:
    """Summarize in which months each column appears and with which parquet types."""
    total_periods = schema["period"].nunique()
    summary = (
        schema.groupby("column")
        .agg(
            period_count=("period", "nunique"),
            first_period=("period", "min"),
            last_period=("period", "max"),
            dtypes=("dtype", lambda values: ", ".join(sorted(set(values)))),
        )
        .reset_index()
    )
    summary["missing_period_count"] = total_periods - summary["period_count"]
    summary["present_in_all_periods"] = summary["missing_period_count"].eq(0)
    return summary.sort_values(["present_in_all_periods", "column"], ascending=[True, True]).reset_index(drop=True)


def load_green_taxi_data(
    files: pd.DataFrame,
    rows_per_file: int | None = None,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load full data or a deterministic first-N-row sample per monthly file."""
    frames: list[pd.DataFrame] = []
    requested_columns = list(columns) if columns is not None else None

    for record in files.to_dict("records"):
        frame = pd.read_parquet(record["path"], columns=requested_columns)
        if rows_per_file is not None:
            frame = frame.head(rows_per_file)
        frame = frame.copy()
        frame["source_year"] = record["year"]
        frame["source_month"] = record["month"]
        frame["source_period"] = record["period"]
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def add_trip_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add profiling-only columns derived from pickup/dropoff timestamps."""
    result = df.copy()
    pickup = pd.to_datetime(result.get("lpep_pickup_datetime"), errors="coerce")
    dropoff = pd.to_datetime(result.get("lpep_dropoff_datetime"), errors="coerce")
    result["trip_duration_minutes"] = (dropoff - pickup).dt.total_seconds() / 60
    result["pickup_date"] = pickup.dt.date
    result["pickup_hour"] = pickup.dt.hour
    return result


def monthly_temporal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize monthly row counts and observed timestamp ranges."""
    result = df.copy()
    result["lpep_pickup_datetime"] = pd.to_datetime(result.get("lpep_pickup_datetime"), errors="coerce")
    result["lpep_dropoff_datetime"] = pd.to_datetime(result.get("lpep_dropoff_datetime"), errors="coerce")
    return (
        result.groupby("source_period")
        .agg(
            rows=("source_period", "size"),
            pickup_min=("lpep_pickup_datetime", "min"),
            pickup_max=("lpep_pickup_datetime", "max"),
            dropoff_min=("lpep_dropoff_datetime", "min"),
            dropoff_max=("lpep_dropoff_datetime", "max"),
        )
        .reset_index()
    )


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages for each dataframe column."""
    rows = len(df)
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing_count": df.isna().sum().to_numpy(),
        }
    )
    summary["missing_pct"] = (summary["missing_count"] / rows * 100).round(3) if rows else 0.0
    return summary.sort_values(["missing_pct", "column"], ascending=[False, True]).reset_index(drop=True)


def quality_check_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute observed data-quality signals without enforcing final thresholds."""
    result = add_trip_derived_columns(df)
    rows = len(result)

    checks = [
        (
            "duplicate_full_rows",
            result.duplicated().sum(),
            "observed_fact",
            "Exact duplicate records across all loaded columns.",
        ),
        (
            "missing_pickup_datetime",
            result["lpep_pickup_datetime"].isna().sum(),
            "candidate_expectation",
            "Pickup timestamp is required for any time-aware target or split.",
        ),
        (
            "missing_dropoff_datetime",
            result["lpep_dropoff_datetime"].isna().sum(),
            "candidate_expectation",
            "Dropoff timestamp is needed for duration and post-trip validation.",
        ),
        (
            "non_positive_duration",
            result["trip_duration_minutes"].le(0).sum(),
            "candidate_expectation",
            "Trips with non-positive duration are invalid-looking for duration modeling.",
        ),
        (
            "duration_over_24_hours",
            result["trip_duration_minutes"].gt(24 * 60).sum(),
            "deferred_decision",
            "Very long trips need review before setting an outlier threshold.",
        ),
        (
            "negative_trip_distance",
            result.get("trip_distance", pd.Series(dtype=float)).lt(0).sum(),
            "candidate_expectation",
            "Negative distance is invalid-looking.",
        ),
        (
            "zero_trip_distance",
            result.get("trip_distance", pd.Series(dtype=float)).eq(0).sum(),
            "deferred_decision",
            "Zero-distance trips may be invalid or special cases; defer filtering policy.",
        ),
        (
            "negative_fare_amount",
            result.get("fare_amount", pd.Series(dtype=float)).lt(0).sum(),
            "deferred_decision",
            "Negative fares may reflect adjustments; defer policy until target selection.",
        ),
        (
            "negative_total_amount",
            result.get("total_amount", pd.Series(dtype=float)).lt(0).sum(),
            "deferred_decision",
            "Negative totals may reflect adjustments; defer policy until target selection.",
        ),
        (
            "missing_pickup_location",
            result.get("PULocationID", pd.Series(dtype="object")).isna().sum(),
            "candidate_expectation",
            "Pickup zone is expected for spatial features and serving examples.",
        ),
        (
            "missing_dropoff_location",
            result.get("DOLocationID", pd.Series(dtype="object")).isna().sum(),
            "candidate_expectation",
            "Dropoff zone is expected for spatial features and serving examples.",
        ),
        (
            "passenger_count_missing",
            result.get("passenger_count", pd.Series(dtype="object")).isna().sum(),
            "deferred_decision",
            "Passenger count availability and imputation policy need profiling evidence.",
        ),
        (
            "passenger_count_zero_or_negative",
            result.get("passenger_count", pd.Series(dtype=float)).le(0).sum(),
            "deferred_decision",
            "Passenger count rules depend on target and serving assumptions.",
        ),
    ]

    summary = pd.DataFrame(checks, columns=["check", "rows", "finding_type", "notes"])
    summary["total_rows"] = rows
    summary["pct_rows"] = (summary["rows"] / rows * 100).round(3) if rows else 0.0
    return summary


def exploratory_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize core numeric trip fields for profiling."""
    result = add_trip_derived_columns(df)
    numeric_columns = [
        column
        for column in [
            "trip_duration_minutes",
            "trip_distance",
            "fare_amount",
            "total_amount",
            "passenger_count",
            "PULocationID",
            "DOLocationID",
        ]
        if column in result.columns
    ]
    return result[numeric_columns].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T.reset_index(names="column")


def candidate_target_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Assess candidate target availability without selecting the final target."""
    result = add_trip_derived_columns(df)
    candidates = [
        (
            "trip_duration_minutes",
            result["trip_duration_minutes"].gt(0),
            "Regression target from pickup/dropoff timestamps; not known before trip completion.",
        ),
        (
            "fare_amount",
            result.get("fare_amount", pd.Series(dtype=float)).notna(),
            "Regression target from metered fare; post-trip field.",
        ),
        (
            "total_amount",
            result.get("total_amount", pd.Series(dtype=float)).notna(),
            "Regression target for total paid amount; includes post-trip components.",
        ),
        (
            "long_trip_indicator",
            result["trip_duration_minutes"].gt(result["trip_duration_minutes"].quantile(0.75)),
            "Classification target candidate based on duration quantile; threshold is exploratory only.",
        ),
        (
            "high_fare_indicator",
            result.get("fare_amount", pd.Series(dtype=float)).gt(result.get("fare_amount", pd.Series(dtype=float)).quantile(0.75)),
            "Classification target candidate based on fare quantile; threshold is exploratory only.",
        ),
    ]

    rows = len(result)
    summary_rows = []
    for name, valid_mask, notes in candidates:
        valid_count = int(valid_mask.fillna(False).sum())
        summary_rows.append(
            {
                "candidate_target": name,
                "usable_rows": valid_count,
                "usable_pct": round(valid_count / rows * 100, 3) if rows else 0.0,
                "decision_status": "deferred",
                "notes": notes,
            }
        )
    return pd.DataFrame(summary_rows)


def serving_time_column_classification(columns: Iterable[str]) -> pd.DataFrame:
    """Classify columns by whether they are likely available at serving time."""
    before_or_at_request = {
        "lpep_pickup_datetime",
        "PULocationID",
        "DOLocationID",
        "passenger_count",
        "RatecodeID",
        "trip_type",
    }
    post_trip = {
        "lpep_dropoff_datetime",
        "trip_distance",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "ehail_fee",
        "improvement_surcharge",
        "total_amount",
        "payment_type",
        "congestion_surcharge",
        "cbd_congestion_fee",
        "trip_duration_minutes",
    }
    operational_or_unknown = {"VendorID", "store_and_fwd_flag"}

    rows = []
    for column in columns:
        if column in before_or_at_request:
            availability = "serving_candidate"
            rationale = "Likely known before or at request time, subject to product design."
        elif column in post_trip:
            availability = "post_trip_only"
            rationale = "Known only after trip completion or payment; avoid as serving input."
        elif column in operational_or_unknown:
            availability = "needs_review"
            rationale = "Operational field; availability depends on integration design."
        elif column.startswith("source_"):
            availability = "metadata_only"
            rationale = "Local profiling metadata, not a model input."
        else:
            availability = "needs_review"
            rationale = "Not classified yet."
        rows.append({"column": column, "availability": availability, "rationale": rationale})
    return pd.DataFrame(rows)


def save_artifacts(artifacts: dict[str, pd.DataFrame], output_dir: str | Path) -> dict[str, str]:
    """Save named dataframe artifacts as CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for name, frame in artifacts.items():
        path = output_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        saved[name] = str(path)
    return saved
