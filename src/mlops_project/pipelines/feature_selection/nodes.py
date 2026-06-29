import logging
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

logger = logging.getLogger(__name__)


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    parameters: dict[str, Any],
) -> list[str]:
    """Select production feature columns from preprocessed training data."""
    if not parameters.get("enabled", False):
        columns = X_train.columns.tolist()
        logger.info("Feature selection disabled; using all %d columns.", len(columns))
        return columns

    estimator = RandomForestRegressor(
        n_estimators=parameters.get("n_estimators", 50),
        max_depth=parameters.get("max_depth", 8),
        random_state=parameters.get("random_state", 42),
        n_jobs=parameters.get("n_jobs", -1),
    )
    n_features_to_select = parameters.get(
        "n_features_to_select",
        max(1, X_train.shape[1] // 2),
    )
    selector = RFE(estimator, n_features_to_select=n_features_to_select)
    selector.fit(X_train, y_train.iloc[:, 0].astype(float))
    columns = X_train.columns[selector.get_support()].tolist()

    logger.info(
        "Feature selection retained %d/%d columns.", len(columns), X_train.shape[1]
    )
    return columns
