import logging
from datetime import datetime

import pandas as pd
import great_expectations as gx
import great_expectations.expectations as gxe

logger = logging.getLogger(__name__)


def _parse_results(results) -> pd.DataFrame:
    rows = []
    for r in results.results:
        kwargs = r.expectation_config.kwargs or {}
        rows.append(
            {
                "success": r.success,
                "expectation_type": r.expectation_config.type,
                "column": kwargs.get("column", ""),
                "min_value": kwargs.get("min_value", ""),
                "max_value": kwargs.get("max_value", ""),
                "value_set": str(kwargs.get("value_set", "")),
                "unexpected_count": r.result.get("unexpected_count", ""),
                "unexpected_percent": r.result.get("unexpected_percent", ""),
                "observed_value": str(r.result.get("observed_value", "")),
            }
        )
    return pd.DataFrame(rows)


def unit_test(ingested_data: pd.DataFrame) -> pd.DataFrame:
    """
    Run GX unit tests on ingested Green Taxi data.
    Expectations are derived from the profiling in notebook 01_data_profiling_and_validation.
    Halts the pipeline if any expectation fails.

    Args:
        ingested_data: Output of the ingestion pipeline.

    Returns:
        reporting_tests: DataFrame with per-expectation validation results.
    """
    logger.info("Starting data unit tests...")

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("kedro_data_source")
    data_asset = data_source.add_dataframe_asset("kedro_data_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe("kedro_batch")

    suite = gx.ExpectationSuite(name="unit_data_tests")

    # Range expectations (from notebook 01 profiling)
    range_rules = {
        "trip_distance":          (0, 100),
        "fare_amount":            (0, 500),
        "extra":                  (0, 50),
        "mta_tax":                (0, 5),
        "tip_amount":             (0, 200),
        "tolls_amount":           (0, 108),
        "improvement_surcharge":  (0, 5),
        "congestion_surcharge":   (0, 20),
        "total_amount":           (0, 600),
        "passenger_count":        (0, 9),
        "lpep_pickup_datetime":   (datetime(2024, 1, 1), datetime(2026, 12, 31, 23, 59, 59)),
        "lpep_dropoff_datetime":  (datetime(2024, 1, 1), datetime(2026, 12, 31, 23, 59, 59)),
    }
    for column, (min_value, max_value) in range_rules.items():
        if column in ingested_data.columns:
            suite.add_expectation(
                gxe.ExpectColumnValuesToBeBetween(
                    column=column, min_value=min_value, max_value=max_value
                )
            )

    # Not null
    for column in ["lpep_pickup_datetime", "lpep_dropoff_datetime", "PULocationID", "DOLocationID"]:
        if column in ingested_data.columns:
            suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=column))

    # Categorical sets
    if "RatecodeID" in ingested_data.columns:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(
                column="RatecodeID", value_set=[1, 2, 3, 4, 5, 6, 99], mostly=0.99
            )
        )
    if "payment_type" in ingested_data.columns:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(
                column="payment_type", value_set=[1.0, 2.0, 3.0, 4.0, 5.0], mostly=0.99
            )
        )

    context.suites.add(suite)

    validation_definition = gx.ValidationDefinition(
        name="kedro_validation", data=batch_def, suite=suite
    )
    context.validation_definitions.add(validation_definition)

    results = validation_definition.run(batch_parameters={"dataframe": ingested_data})

    if not results.success:
        logger.error("Data validation FAILED:")
        for r in results.results:
            if not r.success:
                logger.error("  Failed: %s", r.expectation_config.type)
        raise ValueError("Pipeline halted due to data validation failure.")

    logger.info("All data unit tests passed.")
    return _parse_results(results)
