import logging
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)

logger = logging.getLogger(__name__)

MLFLOW_MODEL_SERIALIZATION_FORMAT = mlflow_sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE


def _evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _validation_predictions(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    actual = y_true.reset_index(drop=True)
    return pd.DataFrame(
        {
            "actual_tip_amount": actual,
            "predicted_tip_amount": y_pred,
            "residual": actual - y_pred,
        }
    )


def _model_params(
    selected_model_metadata: dict[str, Any],
    model_train_parameters: dict[str, Any],
    model_selection_parameters: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    selected_name = selected_model_metadata.get(
        "selected_model_name",
        model_train_parameters.get("selected_model_name", "random_forest_current"),
    )
    candidates = model_selection_parameters.get("candidates", {})
    params = candidates.get(selected_name, model_train_parameters["model_params"])
    return selected_name, params


def _feature_importance_plot(
    model: RandomForestRegressor,
    columns: list[str],
    max_display: int = 20,
):
    importances = pd.Series(model.feature_importances_, index=columns)
    top_importances = importances.sort_values(ascending=False).head(max_display)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_importances.sort_values().plot.barh(ax=ax)
    ax.set_title("Top Random Forest Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def model_train(  # noqa: PLR0913
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    best_columns: list[str],
    selected_model_metadata: dict[str, Any],
    model_train_parameters: dict[str, Any],
    model_selection_parameters: dict[str, Any],
):
    """Train the production estimator on already preprocessed features."""
    columns = [column for column in best_columns if column in X_train.columns]
    if not columns:
        raise ValueError("No selected columns are available for model training.")

    selected_name, params = _model_params(
        selected_model_metadata,
        model_train_parameters,
        model_selection_parameters,
    )
    X_train_selected = X_train.loc[:, columns]
    X_val_selected = X_val.loc[:, columns]
    y_train_series = y_train.iloc[:, 0].astype(float)
    y_val_series = y_val.iloc[:, 0].astype(float)

    model = RandomForestRegressor(**params)

    mlflow.set_tracking_uri(model_train_parameters["mlflow_tracking_uri"])
    mlflow.set_experiment(model_train_parameters["mlflow_experiment_name"])

    with mlflow.start_run(run_name=f"kedro_{selected_name}", nested=True):
        model.fit(X_train_selected, y_train_series)
        y_pred = np.asarray(model.predict(X_val_selected), dtype=float)
        metrics = _evaluate_predictions(y_val_series, y_pred)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "project": "green_taxi_mlops",
                "target": model_train_parameters["target_col"],
                "task": "regression",
                "stage": "kedro_model_train",
                "selected_model_name": selected_name,
                "model_serialization_format": MLFLOW_MODEL_SERIALIZATION_FORMAT,
            }
        )
        mlflow_sklearn.log_model(
            model,
            name="model",
            serialization_format=MLFLOW_MODEL_SERIALIZATION_FORMAT,
        )

    predictions = _validation_predictions(y_val_series, y_pred)
    plot = _feature_importance_plot(model, columns)
    logger.info("Validation RMSE: %.4f", metrics["rmse"])
    return model, metrics, predictions, plot
