import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
import shap
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
REPORTING_DIR = Path("data/08_reporting")
FEATURE_IMPORTANCE_PLOT_PATH = REPORTING_DIR / "feature_importance.png"
FEATURE_IMPORTANCE_TABLE_PATH = REPORTING_DIR / "feature_importance.csv"
SHAP_SUMMARY_PLOT_PATH = REPORTING_DIR / "shap_summary.png"
SHAP_SUMMARY_TABLE_PATH = REPORTING_DIR / "shap_summary.csv"
MULTI_OUTPUT_SHAP_DIMENSIONS = 3
TREE_SHAP_MODEL_TYPES = {
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
}
LINEAR_SHAP_MODEL_TYPES = {
    "linear_regression",
    "ridge",
    "elastic_net",
}


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


def _sample_frame(
    data: pd.DataFrame,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    sample_rows = min(sample_size, len(data))
    if sample_rows <= 0:
        return data.iloc[0:0, :].copy()
    if sample_rows == len(data):
        sample = data
    else:
        sample = data.sample(n=sample_rows, random_state=random_state)
    return sample.reset_index(drop=True)


def _dense_numeric_frame(data: pd.DataFrame) -> pd.DataFrame:
    dense = np.asarray(data, dtype=float)
    return pd.DataFrame(dense, columns=data.columns)


def _clear_conditional_explainability_artifacts() -> None:
    for path in (
        FEATURE_IMPORTANCE_PLOT_PATH,
        FEATURE_IMPORTANCE_TABLE_PATH,
        SHAP_SUMMARY_PLOT_PATH,
        SHAP_SUMMARY_TABLE_PATH,
    ):
        path.unlink(missing_ok=True)


def _artifact_path(path: Path) -> str:
    return path.as_posix()


def _save_feature_importance(
    model: Any,
    columns: list[str],
    parameters: dict[str, Any],
    selected_type: str,
) -> dict[str, Any]:
    max_display = parameters.get("feature_importance_max_display", 20)
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        title = "Top Feature Importances"
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_).astype(float))
        title = "Top Absolute Coefficients"
    else:
        return {
            "generated": False,
            "method": None,
            "skip_reason": (
                f"Selected model type '{selected_type}' does not expose "
                "feature_importances_ or coef_."
            ),
            "plot_path": None,
            "table_path": None,
        }

    if len(values) != len(columns):
        raise ValueError("Feature importance length does not match selected columns.")

    importances = pd.DataFrame(
        {
            "feature": columns,
            "importance": values,
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(FEATURE_IMPORTANCE_TABLE_PATH, index=False)
    top_importances = importances.head(max_display).set_index("feature")["importance"]

    fig, ax = plt.subplots(figsize=(10, 6))
    top_importances.sort_values().plot.barh(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PLOT_PATH, dpi=150)
    plt.close(fig)

    return {
        "generated": True,
        "method": "feature_importances_"
        if hasattr(model, "feature_importances_")
        else "absolute_coef_",
        "skip_reason": None,
        "plot_path": _artifact_path(FEATURE_IMPORTANCE_PLOT_PATH),
        "table_path": _artifact_path(FEATURE_IMPORTANCE_TABLE_PATH),
    }


def _save_shap_explainability(
    model: Any,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    parameters: dict[str, Any],
    selected_type: str,
) -> dict[str, Any]:
    random_state = parameters.get("explainability_random_state", 42)
    shap_sample_size = parameters.get("shap_sample_size", 25)
    background_sample_size = parameters.get("shap_background_sample_size", 100)
    max_display = parameters.get("shap_max_display", 20)

    X_explain = _dense_numeric_frame(
        _sample_frame(X_val, shap_sample_size, random_state)
    )
    if X_explain.empty:
        return {
            "generated": False,
            "method": None,
            "skip_reason": "No validation rows are available for SHAP.",
            "plot_path": None,
            "table_path": None,
            "sample_size": 0,
            "background_sample_size": 0,
        }

    if selected_type in TREE_SHAP_MODEL_TYPES:
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X_explain)
        method = "tree_explainer"
        background_rows = 0
    elif selected_type in LINEAR_SHAP_MODEL_TYPES:
        background = _dense_numeric_frame(
            _sample_frame(X_train, background_sample_size, random_state)
        )
        explainer = shap.LinearExplainer(model, background)
        explanation = explainer(X_explain)
        method = "linear_explainer"
        background_rows = len(background)
    else:
        return {
            "generated": False,
            "method": None,
            "skip_reason": f"Selected model type '{selected_type}' is not configured for SHAP.",
            "plot_path": None,
            "table_path": None,
            "sample_size": len(X_explain),
            "background_sample_size": 0,
        }

    shap_values = np.asarray(explanation.values, dtype=float)
    if shap_values.ndim == MULTI_OUTPUT_SHAP_DIMENSIONS:
        shap_values = shap_values[:, :, 0]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_summary = pd.DataFrame(
        {
            "feature": X_explain.columns,
            "mean_absolute_shap": mean_abs_shap,
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    shap_summary.to_csv(SHAP_SUMMARY_TABLE_PATH, index=False)

    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(SHAP_SUMMARY_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "generated": True,
        "method": method,
        "skip_reason": None,
        "plot_path": _artifact_path(SHAP_SUMMARY_PLOT_PATH),
        "table_path": _artifact_path(SHAP_SUMMARY_TABLE_PATH),
        "sample_size": len(X_explain),
        "background_sample_size": background_rows,
    }


def _save_explainability_artifacts(  # noqa: PLR0913
    model: Any,
    columns: list[str],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    parameters: dict[str, Any],
    selected_name: str,
    selected_type: str,
) -> dict[str, Any]:
    _clear_conditional_explainability_artifacts()
    REPORTING_DIR.mkdir(parents=True, exist_ok=True)

    feature_importance = _save_feature_importance(
        model,
        columns,
        parameters,
        selected_type,
    )
    shap_metadata = _save_shap_explainability(
        model,
        X_train,
        X_val,
        parameters,
        selected_type,
    )
    return {
        "selected_model_name": selected_name,
        "selected_model_type": selected_type,
        "feature_importance": feature_importance,
        "shap": shap_metadata,
    }


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
        try:
            mlflow_sklearn.log_model(
                model,
                name="model",
                serialization_format=MLFLOW_MODEL_SERIALIZATION_FORMAT,
            )
        except Exception as exc:
            logger.warning("Could not log model artifact to MLflow: %s", exc)

    predictions = _validation_predictions(y_val_series, y_pred)
    explainability_metadata = _save_explainability_artifacts(
        model,
        columns,
        X_train_selected,
        X_val_selected,
        model_train_parameters,
        selected_name,
        selected_type,
    )
    logger.info(
        "Train RMSE: %.4f; validation RMSE: %.4f",
        metrics["train_metrics"]["rmse"],
        metrics["validation_metrics"]["rmse"],
    )
    return model, metrics, predictions, explainability_metadata
