import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def split_by_year(ingested_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the ingested dataset into reference (training) and analysis (batch/drift) sets.

    - Reference: 2024 trips  → used for training the model
    - Analysis:  2025+ trips → used for batch inference and drift monitoring

    Args:
        ingested_data: Full combined Green Taxi DataFrame from ingestion pipeline.

    Returns:
        ref_data: 2024 trips (reference / training set).
        ana_data: 2025+ trips (analysis / batch set).
    """
    pickup = pd.to_datetime(ingested_data["lpep_pickup_datetime"])
    year = pickup.dt.year

    ref_data = ingested_data[year == 2024].reset_index(drop=True)
    ana_data = ingested_data[year >= 2025].reset_index(drop=True)

    logger.info("Reference (2024):  %d rows", len(ref_data))
    logger.info("Analysis  (2025+): %d rows", len(ana_data))

    return ref_data, ana_data
