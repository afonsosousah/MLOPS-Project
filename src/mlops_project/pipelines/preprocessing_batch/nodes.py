import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def preprocessing_batch(
    test_data: pd.DataFrame,
    preprocessing_transformer: Any,
    production_columns: Sequence[str],
) -> pd.DataFrame:
    """Apply the fitted preprocessing transformer to held-out batch data."""
    transformed = preprocessing_transformer.transform(test_data).reset_index(drop=True)
    missing_columns = [
        column for column in production_columns if column not in transformed
    ]
    if missing_columns:
        raise ValueError(
            "Batch preprocessing did not create expected production columns: "
            f"{missing_columns[:10]}"
        )

    transformed = transformed.loc[:, list(production_columns)]
    logger.info(
        "Batch preprocessing transformed %d rows with %d production columns.",
        len(transformed),
        len(production_columns),
    )
    return transformed
