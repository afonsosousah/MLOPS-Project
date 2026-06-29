import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def model_predict(
    data: pd.DataFrame,
    model: Any,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run held-out batch predictions with the production model."""
    feature_cols = [column for column in columns if column in data.columns]
    if not feature_cols:
        raise ValueError("No production model columns were found in the batch data.")

    predictions = np.asarray(model.predict(data.loc[:, feature_cols]), dtype=float)
    result = data.copy()
    result["predicted_tip_amount"] = predictions
    summary = {
        "rows": int(len(result)),
        "prediction_mean": float(predictions.mean()),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
    }

    logger.info(
        "Batch prediction complete: %d rows, mean predicted tip = %.2f.",
        len(result),
        summary["prediction_mean"],
    )
    return result, summary
