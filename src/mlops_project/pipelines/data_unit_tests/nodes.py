import logging
from typing import Dict

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
    Run GX unit tests on the ingested Green Taxi data.
    Halts the pipeline if any expectation fails.

    Args:
        ingested_data: Output of the ingestion pipeline.

    Returns:
        reporting_tests: DataFrame with per-expectation validation results.
    """
    logger.info("Running data unit tests...")

    context = gx.get_context(mode="ephemeral")
    ds = context.data_sources.add_pandas("unit_test_source")
    asset = ds.add_dataframe_asset("unit_test_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("batch")

    suite = gx.ExpectationSuite(name="green_taxi_unit_tests")

    # Schema checks
    suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="trip_distance", type_="float64"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="fare_amount", type_="float64"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="PULocationID", type_="int32"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="DOLocationID", type_="int32"))

    # Value range checks
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="trip_distance", min_value=0, max_value=500)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="fare_amount", min_value=-10, max_value=1000)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="passenger_count", min_value=0, max_value=9, mostly=0.99)
    )

    # Not null checks
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="lpep_pickup_datetime"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="PULocationID"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="DOLocationID"))

    # Categorical checks
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="payment_type", value_set=[1.0, 2.0, 3.0, 4.0, 5.0], mostly=0.99
        )
    )

    context.suites.add(suite)

    validation_def = gx.ValidationDefinition(
        name="green_taxi_unit_validation",
        data=batch_def,
        suite=suite,
    )
    context.validation_definitions.add(validation_def)

    results = validation_def.run(batch_parameters={"dataframe": ingested_data})

    if not results.success:
        for r in results.results:
            if not r.success:
                logger.error("Failed: %s on column '%s'",
                             r.expectation_config.type,
                             r.expectation_config.kwargs.get("column", ""))
        raise ValueError("Data unit tests failed. Pipeline halted.")

    logger.info("All data unit tests passed.")
    return _parse_results(results)
