import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import great_expectations as gx
import great_expectations.expectations as gxe
from datetime import datetime

def unit_test(
    data: pd.DataFrame,
): 

    """
    Kedro node to validate incoming data using Great Expectations.
    Halts the pipeline and logs errors if validation fails.
    """
    log = logging.getLogger(__name__)
    log.info("Starting data validation...")

    # Initialize GX Context and Data Source
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("kedro_data_source")
    data_asset = data_source.add_dataframe_asset("kedro_data_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe("kedro_batch")

    # Define the Expectation Suite
    suite = gx.ExpectationSuite(name="unit_data_tests")
    
    # Expectations based on the findings from data profiling
    range_rules = {
        "trip_distance": (0, 100),
        "fare_amount": (0, 500),
        "extra": (0, 50),
        "mta_tax": (0, 5),
        "tip_amount": (0, 200),
        "tolls_amount": (0, 108),
        "improvement_surcharge": (0, 5),
        "congestion_surcharge": (0, 20),
        "total_amount": (0, 600),
        "passenger_count": (0, 9),
        
        "lpep_pickup_datetime": (
            datetime(2024, 1, 1, 0, 0, 0),
            datetime(2026, 4, 30, 23, 59, 59)
        ),
        
        "lpep_dropoff_datetime": (
            datetime(2024, 1, 1, 0, 0, 0),
            datetime(2026, 4, 30, 23, 59, 59)
        )
    }

    for column, (min_value, max_value) in range_rules.items():
        if column in data.columns:
            suite.add_expectation(
                gxe.ExpectColumnValuesToBeBetween(
                    column=column,
                    min_value=min_value,
                    max_value=max_value,
                )
            )

    for column in ["lpep_pickup_datetime", "lpep_dropoff_datetime", "PULocationID", "DOLocationID"]:
        if column in data.columns:
            suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=column))

    if "RatecodeID" in data.columns:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(
                column="RatecodeID",
                value_set=[1, 2, 3, 4, 5, 6, 99],
                mostly=0.99,
            )
        )
    
    if "payment_type" in data.columns:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(
                column="payment_type",
                value_set=[1, 2, 3, 4, 5, 6],
                mostly=0.99,
            )
        )

    context.suites.add(suite)

    # Define and run the validation
    validation_definition = gx.ValidationDefinition(
        name="kedro_validation",
        data=batch_def,
        suite=suite,
    )
    context.validation_definitions.add(validation_definition)

    # Pass the Kedro DataFrame into GX
    results = validation_definition.run(batch_parameters={"dataframe": data})

    # Evaluate Results and Log
    if not results.success:
        log.error("❌ Data validation FAILED. See details below:")
        
        # Loop through and log exactly which expectations failed
        for result in results.results:
            if not result.success:
                failed_config = result.expectation_config.type
                log.error(f"Failed Expectation: {failed_config}")
        
        # Stop the Kedro pipeline from continuing
        raise ValueError("Pipeline halted due to data validation failure.")

    # If everything passes, log success and return the data
    log.info("✅ Data passed on the unit data tests")
    
    return None