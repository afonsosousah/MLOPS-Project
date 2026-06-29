import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

logger = logging.getLogger(__name__)


def _sample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    max_rows: int | None,
) -> tuple[pd.DataFrame, pd.Series]:
    y = y_train.iloc[:, 0].astype(float)
    if max_rows is None or max_rows <= 0 or len(X_train) <= max_rows:
        return X_train, y
    sample = X_train.sample(n=max_rows, random_state=42).index
    return X_train.loc[sample], y.loc[sample]


def model_selection(  # noqa: PLR0913
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    best_columns: list[str],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Compare configured model candidates on validation RMSE."""
    columns = [column for column in best_columns if column in X_train.columns]
    if not columns:
        raise ValueError("No selected columns are available for model selection.")

    if not parameters.get("enabled", True):
        return {
            "enabled": False,
            "selected_model_name": "random_forest_current",
            "selected_columns": columns,
        }

    X_train_selected = X_train.loc[:, columns]
    X_val_selected = X_val.loc[:, columns]
    X_fit, y_fit = _sample_training_data(
        X_train_selected,
        y_train,
        parameters.get("max_rows"),
    )
    y_val_series = y_val.iloc[:, 0].astype(float)

    results = []
    for name, params in parameters.get("candidates", {}).items():
        model = RandomForestRegressor(**params)
        model.fit(X_fit, y_fit)
        predictions = np.asarray(model.predict(X_val_selected), dtype=float)
        metrics = {
            "mae": float(mean_absolute_error(y_val_series, predictions)),
            "rmse": float(root_mean_squared_error(y_val_series, predictions)),
            "r2": float(r2_score(y_val_series, predictions)),
        }
        results.append({"name": name, "params": params, "metrics": metrics})
        logger.info("Candidate %s validation RMSE: %.4f", name, metrics["rmse"])

    if not results:
        raise ValueError("No model_selection candidates were configured.")

    selected = min(results, key=lambda item: item["metrics"]["rmse"])
    logger.info(
        "Selected model candidate %s with validation RMSE %.4f.",
        selected["name"],
        selected["metrics"]["rmse"],
    )
    return {
        "enabled": True,
        "selected_model_name": selected["name"],
        "selected_columns": columns,
        "candidates": results,
    }
