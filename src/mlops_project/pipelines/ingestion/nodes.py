import logging
from typing import Any, Callable, Dict

import pandas as pd
import great_expectations as gx
import great_expectations.expectations as gxe
import hopsworks

logger = logging.getLogger(__name__)

AIRPORT_ZONE_IDS = {1, 132, 138}


# ---------------------------------------------------------------------------
# Hopsworks helper
# ---------------------------------------------------------------------------

def _to_feature_store(data: pd.DataFrame, group_name: str, description: str, credentials: dict):
    project = hopsworks.login(
        api_key_value=credentials["FS_API_KEY"],
        project=credentials["FS_PROJECT_NAME"],
    )
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=group_name,
        version=1,
        description=description,
        primary_key=["index"],
        online_enabled=False,
    )
    fg.insert(
        data.reset_index().rename(columns={"index": "index"}),
        overwrite=False,
        write_options={"wait_for_job": True},
    )
    fg.compute_statistics()
    logger.info("Uploaded %d rows to feature group '%s'", len(data), group_name)


# ---------------------------------------------------------------------------
# GX validation
# ---------------------------------------------------------------------------

def _build_suite(context) -> gx.ExpectationSuite:
    suite = gx.ExpectationSuite(name="green_taxi_ingestion_suite")

    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="lpep_pickup_datetime"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="lpep_dropoff_datetime"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="PULocationID"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="DOLocationID"))
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="trip_distance", min_value=0, max_value=500)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="fare_amount", min_value=-10, max_value=1000)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="payment_type", value_set=[1.0, 2.0, 3.0, 4.0, 5.0], mostly=0.99
        )
    )

    return context.suites.add(suite)


def _validate(df: pd.DataFrame) -> bool:
    context = gx.get_context(mode="ephemeral")
    suite = _build_suite(context)

    ds = context.data_sources.add_pandas("ingestion_source")
    asset = ds.add_dataframe_asset("ingestion_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    result = batch.validate(suite)

    if not result.success:
        for r in result.results:
            if not r.success:
                logger.error("Failed expectation: %s", r.expectation_config.type)
        raise ValueError("Ingested data failed GX validation. Pipeline halted.")

    logger.info("GX validation passed.")
    return True


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def ingestion(
    partitioned_input: Dict[str, Callable],
    zone_lookup: pd.DataFrame,
    parameters: Dict[str, Any],
) -> pd.DataFrame:
    """
    Load all monthly Green Taxi parquet partitions, join zone lookup,
    run GX validation, and optionally push to Hopsworks Feature Store.

    Args:
        partitioned_input: Dict of {partition_id: load_fn} from PartitionedDataset.
        zone_lookup: NYC taxi zone lookup table.
        parameters: Kedro parameters (target_column, to_feature_store, ...).

    Returns:
        ingested_data: Combined and validated DataFrame.
    """
    # 1. Load and concat all partitions
    logger.info("Loading %d parquet partitions...", len(partitioned_input))
    dfs = [loader() for loader in partitioned_input.values()]
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d rows, %d columns.", len(df), len(df.columns))

    # 2. Drop fully null column
    df = df.drop(columns=["ehail_fee"], errors="ignore")

    # 3. Join borough info from zone lookup
    zone_map = zone_lookup.set_index("LocationID")["Borough"].to_dict()
    df["PU_borough"] = df["PULocationID"].map(zone_map).fillna("Unknown")
    df["DO_borough"] = df["DOLocationID"].map(zone_map).fillna("Unknown")

    # 4. GX validation
    logger.info("Running GX validation on ingested data...")
    _validate(df)

    # 5. Optional Hopsworks push
    if parameters.get("to_feature_store", False):
        from kedro.config import OmegaConfigLoader
        from kedro.framework.project import settings
        from pathlib import Path
        conf_loader = OmegaConfigLoader(conf_source=str(Path("") / settings.CONF_SOURCE))
        credentials = conf_loader["credentials"]["feature_store"]

        logger.info("Pushing ingested data to Hopsworks Feature Store...")
        _to_feature_store(df, "green_taxi_ingested", "Raw ingested Green Taxi data", credentials)

    return df
