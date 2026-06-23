import logging

import great_expectations as gx
import pandas as pd

from great_expectations.expectations.core.expect_column_pair_values_a_to_be_greater_than_b import ExpectColumnPairValuesAToBeGreaterThanB
from great_expectations.expectations.core.expect_column_values_to_be_between import ExpectColumnValuesToBeBetween
from great_expectations.expectations.core.expect_column_values_to_be_in_set import ExpectColumnValuesToBeInSet
from great_expectations.expectations.core.expect_column_values_to_not_be_null import ExpectColumnValuesToNotBeNull
from great_expectations.expectations.core.expect_table_columns_to_match_set import ExpectTableColumnsToMatchSet

from .utils import _parse_results

logger = logging.getLogger(__name__)

EXPECTED_BASE_COLUMNS = [
    "VendorID",
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "store_and_fwd_flag",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "payment_type",
    "trip_type",
    "congestion_surcharge",
]

REQUIRED_COLUMNS = [
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "fare_amount",
    "total_amount",
]

CODE_COMPLETENESS_COLUMNS = [
    "RatecodeID",
    "payment_type",
    "trip_type",
    "store_and_fwd_flag",
    "passenger_count",
    "congestion_surcharge",
]

OFFICIAL_CODE_SETS = {
    "RatecodeID": [1, 2, 3, 4, 5, 6, 99],
    "payment_type": [1, 2, 3, 4, 5, 6],
    "trip_type": [1, 2],
    "store_and_fwd_flag": ["Y", "N"],
}

CODE_COMPLETENESS_MOSTLY = 0.90
COMMON_ANOMALY_MOSTLY = 0.99


def _add_if_present(
    suite: gx.ExpectationSuite,
    df: pd.DataFrame,
    column: str,
    expectation,
) -> None:
    if column in df.columns:
        suite.add_expectation(expectation)


def _build_suite(ingested_data: pd.DataFrame) -> gx.ExpectationSuite:
    suite = gx.ExpectationSuite(name="green_taxi_data_unit_tests")

    suite.add_expectation(
        ExpectTableColumnsToMatchSet(
            column_set=EXPECTED_BASE_COLUMNS,
            exact_match=False,
        )
    )

    for column in REQUIRED_COLUMNS:
        _add_if_present(
            suite,
            ingested_data,
            column,
            ExpectColumnValuesToNotBeNull(column=column)
        )

    _add_if_present(
        suite,
        ingested_data,
        "PULocationID",
        ExpectColumnValuesToBeBetween(column="PULocationID", min_value=1, max_value=265),
    )
    _add_if_present(
        suite,
        ingested_data,
        "DOLocationID",
        ExpectColumnValuesToBeBetween(column="DOLocationID", min_value=1, max_value=265),
    )

    for column in CODE_COMPLETENESS_COLUMNS:
        _add_if_present(
            suite,
            ingested_data,
            column,
            ExpectColumnValuesToNotBeNull(
                column=column,
                mostly=CODE_COMPLETENESS_MOSTLY,
            ),
        )

    for column, value_set in OFFICIAL_CODE_SETS.items():
        _add_if_present(
            suite,
            ingested_data,
            column,
            ExpectColumnValuesToBeInSet(
                column=column,
                value_set=value_set,
                mostly=CODE_COMPLETENESS_MOSTLY,
            ),
        )

    if {"lpep_dropoff_datetime", "lpep_pickup_datetime"}.issubset(ingested_data.columns):
        suite.add_expectation(
            ExpectColumnPairValuesAToBeGreaterThanB(
                column_A="lpep_dropoff_datetime",
                column_B="lpep_pickup_datetime",
                mostly=COMMON_ANOMALY_MOSTLY,
            )
        )

    _add_if_present(
        suite,
        ingested_data,
        "trip_distance",
        ExpectColumnValuesToBeBetween(column="trip_distance", min_value=0),
    )
    _add_if_present(
        suite,
        ingested_data,
        "fare_amount",
        ExpectColumnValuesToBeBetween(
            column="fare_amount",
            min_value=0,
            mostly=COMMON_ANOMALY_MOSTLY,
        ),
    )
    _add_if_present(
        suite,
        ingested_data,
        "total_amount",
        ExpectColumnValuesToBeBetween(
            column="total_amount",
            min_value=0,
            mostly=COMMON_ANOMALY_MOSTLY,
        ),
    )

    return suite


def unit_test(ingested_data: pd.DataFrame) -> pd.DataFrame:
    """
    Run profiling-based Great Expectations checks on ingested Green Taxi data.

    The checks come from Notebook 1 and live in this validation pipeline rather than
    in ingestion, so ingestion stays limited to loading and stable enrichment.
    """
    logger.info("Starting data unit tests...")

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("kedro_data_source")
    data_asset = data_source.add_dataframe_asset("kedro_data_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe("kedro_batch")

    suite = _build_suite(ingested_data)
    context.suites.add(suite)

    validation_definition = gx.ValidationDefinition(
        name="kedro_validation",
        data=batch_def,
        suite=suite,
    )
    context.validation_definitions.add(validation_definition)

    results = validation_definition.run(batch_parameters={"dataframe": ingested_data})
    report = _parse_results(results)

    if not results.success:
        failed = report[~report["success"]]
        logger.error("Data validation failed with %d failed expectations.", len(failed))
        for _, row in failed.iterrows():
            logger.error(
                "Failed: %s on %s",
                row["expectation_type"],
                row["column"] or "table",
            )
        raise ValueError("Pipeline halted due to data validation failure.")

    logger.info("All data unit tests passed.")
    return report
