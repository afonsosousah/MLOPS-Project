import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from mlops_project.pipelines.model_selection.nodes import (
    build_regressor,
    candidate_type_and_params,
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


def _combined_metrics(
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        **validation_metrics,
        **{f"train_{name}": value for name, value in train_metrics.items()},
        **{f"validation_{name}": value for name, value in validation_metrics.items()},
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
    }


def _mlflow_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {name: value for name, value in metrics.items() if isinstance(value, float)}


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
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    selected_name = selected_model_metadata.get(
        "selected_model_name",
        model_train_parameters.get("selected_model_name", "random_forest"),
    )
    if "selected_model_params" in selected_model_metadata:
        model_type = selected_model_metadata.get("selected_model_type", "random_forest")
        params = dict(selected_model_metadata["selected_model_params"])
        candidate = {"type": model_type, "params": params}
        return selected_name, model_type, params, candidate

    candidates = model_selection_parameters.get("candidates", {})
    fallback_candidate = {
        "type": selected_model_metadata.get("selected_model_type", "random_forest"),
        "params": model_train_parameters["model_params"],
    }
    candidate = candidates.get(selected_name, fallback_candidate)
    model_type, params = candidate_type_and_params(candidate)
    return selected_name, model_type, params, candidate


def _feature_importance_plot(
    model: Any,
    columns: list[str],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    parameters: dict[str, Any],
):
    max_display = parameters.get("feature_importance_max_display", 20)
    sample_size = parameters.get("feature_importance_sample_size", 1000)
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        title = "Top Feature Importances"
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_).astype(float))
        title = "Top Absolute Coefficients"
    else:
        sample_rows = min(sample_size, len(X_val))
        X_sample = X_val.iloc[:sample_rows, :]
        y_sample = y_val.iloc[:sample_rows]
        result = permutation_importance(
            model,
            X_sample,
            y_sample,
            n_repeats=5,
            random_state=42,
            scoring="neg_root_mean_squared_error",
        )
        values = np.asarray(result.importances_mean, dtype=float)
        title = "Top Permutation Importances"

    if len(values) != len(columns):
        raise ValueError("Feature importance length does not match selected columns.")

    importances = pd.Series(values, index=columns)
    top_importances = importances.sort_values(ascending=False).head(max_display)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_importances.sort_values().plot.barh(ax=ax)
    ax.set_title(title)
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
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
):
    """Train the production estimator on already preprocessed features."""
    columns = [column for column in best_columns if column in X_train.columns]
    if not columns:
        raise ValueError("No selected columns are available for model training.")

    selected_name, selected_type, params, candidate = _model_params(
        selected_model_metadata,
        model_train_parameters,
        model_selection_parameters,
    )
    X_train_selected = X_train.loc[:, columns]
    X_val_selected = X_val.loc[:, columns]
    y_train_series = y_train.iloc[:, 0].astype(float)
    y_val_series = y_val.iloc[:, 0].astype(float)

    model = build_regressor(candidate)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run(run_name=f"kedro_{selected_name}", nested=True):
        model.fit(X_train_selected, y_train_series)
        train_pred = np.asarray(model.predict(X_train_selected), dtype=float)
        y_pred = np.asarray(model.predict(X_val_selected), dtype=float)
        metrics = _combined_metrics(
            _evaluate_predictions(y_train_series, train_pred),
            _evaluate_predictions(y_val_series, y_pred),
        )
        mlflow.log_params(params)
        mlflow.log_metrics(_mlflow_metrics(metrics))
        mlflow.set_tags(
            {
                "project": "green_taxi_mlops",
                "target": model_train_parameters["target_col"],
                "task": "regression",
                "stage": "kedro_model_train",
                "selected_model_name": selected_name,
                "selected_model_type": selected_type,
                "model_serialization_format": MLFLOW_MODEL_SERIALIZATION_FORMAT,
            }
        )
        mlflow_sklearn.log_model(
            model,
            name="model",
            serialization_format=MLFLOW_MODEL_SERIALIZATION_FORMAT,
        )

    predictions = _validation_predictions(y_val_series, y_pred)
    plot = _feature_importance_plot(
        model,
        columns,
        X_val_selected,
        y_val_series,
        model_train_parameters,
    )
    logger.info(
        "Train RMSE: %.4f; validation RMSE: %.4f",
        metrics["train_metrics"]["rmse"],
        metrics["validation_metrics"]["rmse"],
    )
    return model, metrics, predictions, plot
