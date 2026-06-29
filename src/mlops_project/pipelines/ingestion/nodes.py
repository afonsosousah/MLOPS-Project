import logging
import re
from collections.abc import Callable

import pandas as pd

logger = logging.getLogger(__name__)

PARTITION_YEAR_PATTERN = re.compile(r"20\d{2}")


def _partition_year(partition_id: str) -> int:
    match = PARTITION_YEAR_PATTERN.search(partition_id)
    if match is None:
        raise ValueError(f"Could not infer year from partition id: {partition_id}")
    return int(match.group())


def _load_partitions_for_years(
    partitioned_input: dict[str, Callable[[], pd.DataFrame]],
    years: list[int],
) -> pd.DataFrame:
    requested_years = set(years)
    selected_loaders = {
        partition_id: loader
        for partition_id, loader in partitioned_input.items()
        if _partition_year(partition_id) in requested_years
    }

    if not selected_loaders:
        available_years = sorted(
            {_partition_year(partition_id) for partition_id in partitioned_input}
        )
        raise ValueError(
            f"No Green Taxi partitions found for years {years}. "
            f"Available years: {available_years}"
        )

    logger.info(
        "Loading %d Green Taxi partitions for years %s.",
        len(selected_loaders),
        years,
    )

    dataframes = []
    for partition_id, loader in sorted(selected_loaders.items()):
        partition = loader()
        partition["source_partition"] = partition_id
        partition["source_year"] = _partition_year(partition_id)
        dataframes.append(partition)

    return pd.concat(dataframes, ignore_index=True)


def _add_borough_columns(
    data: pd.DataFrame,
    zone_lookup: pd.DataFrame,
) -> pd.DataFrame:
    enriched = data.copy()
    zone_map = zone_lookup.set_index("LocationID")["Borough"].to_dict()
    enriched["PU_borough"] = enriched["PULocationID"].map(zone_map).fillna("Unknown")
    enriched["DO_borough"] = enriched["DOLocationID"].map(zone_map).fillna("Unknown")
    return enriched


def ingestion(
    partitioned_input: dict[str, Callable[[], pd.DataFrame]],
    zone_lookup: pd.DataFrame,
    modeling_years: list[int],
    drift_years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Green Taxi partitions, join zone lookup, and separate modeling/drift data.

    Args:
        partitioned_input: Dict of {partition_id: load_fn} from PartitionedDataset.
        zone_lookup: NYC taxi zone lookup table.
        modeling_years: Years used by model development pipelines.
        drift_years: Future years held back for drift monitoring.

    Returns:
        ingested_data: Modeling-period data enriched with borough columns.
        drift_data: Drift holdout data enriched with borough columns.
    """
    ingested_data = _load_partitions_for_years(partitioned_input, modeling_years)
    drift_data = _load_partitions_for_years(partitioned_input, drift_years)

    ingested_data = _add_borough_columns(ingested_data, zone_lookup)
    drift_data = _add_borough_columns(drift_data, zone_lookup)

    logger.info(
        "Created modeling dataset with %d rows and drift dataset with %d rows.",
        len(ingested_data),
        len(drift_data),
    )

    return ingested_data, drift_data
