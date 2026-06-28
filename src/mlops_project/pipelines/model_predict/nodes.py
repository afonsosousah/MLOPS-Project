import logging
from typing import Any, Dict, List

import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def model_predict(
    model: Pipeline,
    columns: List[str],
    data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> pd.DataFrame:
    """Run batch predictions with the production model.

    Selects the columns the model was trained on (dropping any that are absent
    from *data*, e.g. the target), runs inference, and returns the input data
    augmented with a **predicted_tip_amount** column and, when the ground-truth
    target is present, a **residual** column.
    """
    target_col = parameters["target_col"]

    feature_cols = [c for c in columns if c in data.columns and c != target_col]
    X = data[feature_cols]
    y_pred = model.predict(X)

    result = data.copy()
    result["predicted_tip_amount"] = y_pred

    if target_col in data.columns:
        result["residual"] = result[target_col] - y_pred

    logger.info(
        "Batch prediction complete: %d rows, mean predicted tip = %.2f.",
        len(result),
        float(y_pred.mean()),
    )

    return result
