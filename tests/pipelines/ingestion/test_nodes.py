import pandas as pd
import pytest

from mlops_project.pipelines.ingestion.nodes import ingestion


def _green_taxi_partition(pickup_location: int, dropoff_location: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PULocationID": [pickup_location],
            "DOLocationID": [dropoff_location],
        }
    )


def test_ingestion_separates_modeling_and_drift_years() -> None:
    partitioned_input = {
        "2024/green_tripdata_2024-01": lambda: _green_taxi_partition(1, 2),
        "2025/green_tripdata_2025-01": lambda: _green_taxi_partition(2, 3),
        "2026/green_tripdata_2026-01": lambda: _green_taxi_partition(3, 99),
    }
    zone_lookup = pd.DataFrame(
        {
            "LocationID": [1, 2, 3],
            "Borough": ["EWR", "Queens", "Brooklyn"],
        }
    )

    modeling_data, drift_data = ingestion(
        partitioned_input=partitioned_input,
        zone_lookup=zone_lookup,
        modeling_years=[2024, 2025],
        drift_years=[2026],
    )

    assert modeling_data["source_year"].tolist() == [2024, 2025]
    assert drift_data["source_year"].tolist() == [2026]
    assert modeling_data["PU_borough"].tolist() == ["EWR", "Queens"]
    assert drift_data["DO_borough"].tolist() == ["Unknown"]


def test_ingestion_raises_when_configured_year_has_no_partitions() -> None:
    partitioned_input = {
        "2024/green_tripdata_2024-01": lambda: _green_taxi_partition(1, 2),
    }
    zone_lookup = pd.DataFrame({"LocationID": [1, 2], "Borough": ["EWR", "Queens"]})

    with pytest.raises(ValueError, match="No Green Taxi partitions found"):
        ingestion(
            partitioned_input=partitioned_input,
            zone_lookup=zone_lookup,
            modeling_years=[2024],
            drift_years=[2026],
        )
