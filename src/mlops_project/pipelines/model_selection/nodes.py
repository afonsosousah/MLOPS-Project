import logging
from typing import Any

import mlflow
import mlflow.sklearn as mlflow_sklearn
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import RandomSampler, TPESampler
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

logger = logging.getLogger(__name__)

DEFAULT_SELECTION_METRIC = "rmse"
MLFLOW_MODEL_SERIALIZATION_FORMAT = mlflow_sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE

REGRESSOR_TYPES = {
    "dummy": DummyRegressor,
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "elastic_net": ElasticNet,
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
}


def candidate_type_and_params(candidate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the configured regressor type and estimator parameters."""
    model_type = candidate.get("type")
    if model_type is None:
        model_type = "random_forest"
        params = candidate
    else:
        params = candidate.get("params", {})

    if model_type not in REGRESSOR_TYPES:
        supported = ", ".join(sorted(REGRESSOR_TYPES))
        raise ValueError(
            f"Unsupported model_selection candidate type '{model_type}'. "
            f"Supported types are: {supported}."
        )

    return model_type, dict(params)


def build_regressor(candidate: dict[str, Any]):
    """Build an sklearn regressor from a model-selection candidate config."""
    model_type, params = candidate_type_and_params(candidate)
    return REGRESSOR_TYPES[model_type](**params)


def sample_search_space(
    trial: optuna.Trial,
    search_space: dict[str, dict[str, Any]],
    prefix: str = "",
) -> dict[str, Any]:
    """Sample estimator parameters from a YAML-defined Optuna search space."""
    params = {}
    for param_name, spec in search_space.items():
        trial_param_name = f"{prefix}__{param_name}" if prefix else param_name
        search_type = spec.get("type")
        if search_type == "int":
            params[param_name] = trial.suggest_int(
                trial_param_name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        elif search_type == "float":
            params[param_name] = trial.suggest_float(
                trial_param_name,
                float(spec["low"]),
                float(spec["high"]),
                step=spec.get("step"),
                log=bool(spec.get("log", False)),
            )
        elif search_type == "categorical":
            params[param_name] = trial.suggest_categorical(
                trial_param_name,
                spec["choices"],
            )
        else:
            raise ValueError(
                f"Unsupported Optuna search-space type '{search_type}' "
                f"for parameter '{param_name}'."
            )
    return params


def _evaluate_candidate(
    candidate: dict[str, Any],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> dict[str, dict[str, float]]:
    model = build_regressor(candidate)
    model.fit(X_train, y_train)
    train_predictions = np.asarray(model.predict(X_train), dtype=float)
    validation_predictions = np.asarray(model.predict(X_val), dtype=float)
    train_metrics = _regression_metrics(y_train, train_predictions)
    validation_metrics = _regression_metrics(y_val, validation_predictions)
    return {
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "metrics": validation_metrics,
    }


def _regression_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(root_mean_squared_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def _prefixed_metrics(evaluation: dict[str, dict[str, float]]) -> dict[str, float]:
    train_metrics = evaluation["train_metrics"]
    validation_metrics = evaluation["validation_metrics"]
    return {
        **validation_metrics,
        **{f"train_{name}": value for name, value in train_metrics.items()},
        **{f"validation_{name}": value for name, value in validation_metrics.items()},
    }


def _fit_candidate(
    candidate: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    model = build_regressor(candidate)
    model.fit(X_train, y_train)
    return model


def _fixed_candidate_result(
    name: str,
    candidate: dict[str, Any],
    evaluation: dict[str, dict[str, float]],
) -> dict[str, Any]:
    model_type, params = candidate_type_and_params(candidate)
    return {
        "name": name,
        "type": model_type,
        "eligible": bool(candidate.get("eligible", True)),
        "params": params,
        "metrics": evaluation["metrics"],
        "train_metrics": evaluation["train_metrics"],
        "validation_metrics": evaluation["validation_metrics"],
    }


def _eligible_candidates(
    candidates: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        name: candidate
        for name, candidate in candidates.items()
        if bool(candidate.get("eligible", True))
    }


def _build_trial_candidate(
    trial: optuna.Trial,
    candidates: dict[str, dict[str, Any]],
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    candidate_name = trial.suggest_categorical("model_name", list(candidates))
    candidate = candidates[candidate_name]
    model_type, base_params = candidate_type_and_params(candidate)
    trial_params = sample_search_space(
        trial,
        candidate.get("search_space", {}),
        prefix=candidate_name,
    )
    params = {**base_params, **trial_params}
    return candidate_name, model_type, params, {"type": model_type, "params": params}


def _sampler(search_parameters: dict[str, Any]):
    random_state = search_parameters.get("random_state", 42)
    sampler_name = search_parameters.get("sampler", "tpe")
    if sampler_name == "tpe":
        return TPESampler(seed=random_state)
    if sampler_name == "random":
        return RandomSampler(seed=random_state)
    raise ValueError(f"Unsupported Optuna sampler '{sampler_name}'.")


def _trial_result(
    trial: optuna.trial.FrozenTrial,
    model_artifact_logged: bool = False,
) -> dict[str, Any]:
    attrs = trial.user_attrs
    return {
        "number": trial.number,
        "state": trial.state.name,
        "model_name": attrs.get("model_name"),
        "model_type": attrs.get("model_type"),
        "params": attrs.get("params"),
        "metrics": attrs.get("metrics"),
        "train_metrics": attrs.get("train_metrics"),
        "validation_metrics": attrs.get("validation_metrics"),
        "mlflow_run_id": attrs.get("mlflow_run_id"),
        "model_artifact_logged": model_artifact_logged,
    }


def _log_best_trial_model(
    run_id: str,
    candidate: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    model = _fit_candidate(candidate, X_train, y_train)
    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow_sklearn.log_model(
            model,
            name="best_trial_model",
            serialization_format=MLFLOW_MODEL_SERIALIZATION_FORMAT,
        )


def _run_optuna_search(
    candidates: dict[str, dict[str, Any]],
    data: dict[str, pd.DataFrame | pd.Series],
    parameters: dict[str, Any],
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
) -> dict[str, Any]:
    metric_name = parameters.get("metric", DEFAULT_SELECTION_METRIC)
    search_parameters = parameters.get("search", {})
    study = optuna.create_study(
        direction=search_parameters.get("direction", "minimize"),
        sampler=_sampler(search_parameters),
    )
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    def objective(trial: optuna.Trial) -> float:
        candidate_name, model_type, params, trial_candidate = _build_trial_candidate(
            trial,
            candidates,
        )
        with mlflow.start_run(
            run_name=f"optuna_trial_{trial.number}_{candidate_name}",
            nested=True,
        ) as trial_run:
            trial.set_user_attr("mlflow_run_id", trial_run.info.run_id)
            trial.set_user_attr("model_name", candidate_name)
            trial.set_user_attr("model_type", model_type)
            trial.set_user_attr("params", params)
            mlflow.log_params(
                {
                    "model_name": candidate_name,
                    "model_type": model_type,
                    **params,
                }
            )
            mlflow.set_tags(
                {
                    "project": "green_taxi_mlops",
                    "stage": "model_selection_optuna",
                    "trial_number": str(trial.number),
                    "model_name": candidate_name,
                    "model_type": model_type,
                    "selection_metric": metric_name,
                    "trial_state": "RUNNING",
                }
            )
            evaluation = _evaluate_candidate(
                trial_candidate,
                data["X_train"],  # type: ignore
                data["X_val"],  # type: ignore
                data["y_train"],  # type: ignore
                data["y_val"],  # type: ignore
            )
            mlflow.log_metrics(_prefixed_metrics(evaluation))
            mlflow.set_tag("trial_state", "COMPLETE")
            trial.set_user_attr("metrics", evaluation["metrics"])
            trial.set_user_attr("train_metrics", evaluation["train_metrics"])
            trial.set_user_attr("validation_metrics", evaluation["validation_metrics"])
        return evaluation["metrics"][metric_name]

    with mlflow.start_run(run_name="model_selection_optuna", nested=True) as parent_run:
        mlflow.set_tags(
            {
                "project": "green_taxi_mlops",
                "stage": "model_selection_optuna_parent",
                "selection_metric": metric_name,
            }
        )
        mlflow.log_params(
            {
                "optuna_n_trials": int(search_parameters.get("n_trials", 50)),
                "optuna_direction": search_parameters.get("direction", "minimize"),
                "optuna_sampler": search_parameters.get("sampler", "tpe"),
                "optuna_random_state": search_parameters.get("random_state", 42),
            }
        )
        study.optimize(objective, n_trials=int(search_parameters.get("n_trials", 50)))
        parent_run_id = parent_run.info.run_id

    best_trial = study.best_trial
    best_attrs = best_trial.user_attrs
    best_trial_run_id = best_attrs["mlflow_run_id"]
    _log_best_trial_model(
        best_trial_run_id,
        {
            "type": best_attrs["model_type"],
            "params": best_attrs["params"],
        },
        data["X_train"],  # type: ignore
        data["y_train"],  # type: ignore
    )
    best_trial_number = best_trial.number
    return {
        "selected_model_name": best_attrs["model_name"],
        "selected_model_type": best_attrs["model_type"],
        "selected_model_params": best_attrs["params"],
        "selection_source": "optuna",
        "optuna_parent_run_id": parent_run_id,
        "optuna_best_trial": _trial_result(best_trial, model_artifact_logged=True),
        "optuna_trials": [
            _trial_result(
                trial,
                model_artifact_logged=trial.number == best_trial_number,
            )
            for trial in study.trials
        ],
    }


def model_selection(  # noqa: PLR0913
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    best_columns: list[str],
    parameters: dict[str, Any],
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
) -> dict[str, Any]:
    """Compare configured model candidates on validation RMSE."""
    columns = [column for column in best_columns if column in X_train.columns]
    if not columns:
        raise ValueError("No selected columns are available for model selection.")

    if not parameters.get("enabled", True):
        return {
            "enabled": False,
            "selected_model_name": "random_forest",
            "selected_model_type": "random_forest",
            "selected_model_params": {},
            "selected_columns": columns,
            "selection_source": "disabled",
            "selection_metric": parameters.get("metric", DEFAULT_SELECTION_METRIC),
        }

    X_train_selected = X_train.loc[:, columns]
    X_val_selected = X_val.loc[:, columns]
    y_train_series = y_train.iloc[:, 0].astype(float)
    y_val_series = y_val.iloc[:, 0].astype(float)

    fixed_results = []
    candidates = parameters.get("candidates", {})
    metric_name = parameters.get("metric", DEFAULT_SELECTION_METRIC)
    for name, candidate in candidates.items():
        evaluation = _evaluate_candidate(
            candidate,
            X_train_selected,
            X_val_selected,
            y_train_series,
            y_val_series,
        )
        fixed_results.append(_fixed_candidate_result(name, candidate, evaluation))
        logger.info(
            "Candidate %s train RMSE: %.4f; validation RMSE: %.4f",
            name,
            evaluation["train_metrics"]["rmse"],
            evaluation["validation_metrics"]["rmse"],
        )

    if not fixed_results:
        raise ValueError("No model_selection candidates were configured.")

    eligible_candidate_configs = _eligible_candidates(candidates)
    eligible_results = [result for result in fixed_results if result["eligible"]]
    if not eligible_results:
        raise ValueError("No eligible model_selection candidates were configured.")

    if parameters.get("search", {}).get("enabled", False):
        selected_metadata = _run_optuna_search(
            eligible_candidate_configs,
            {
                "X_train": X_train_selected,
                "X_val": X_val_selected,
                "y_train": y_train_series,
                "y_val": y_val_series,
            },
            parameters,
            mlflow_tracking_uri,
            mlflow_experiment_name,
        )
        selected_metric = selected_metadata["optuna_best_trial"]["metrics"][metric_name]
    else:
        selected = min(eligible_results, key=lambda item: item["metrics"][metric_name])
        selected_metadata = {
            "selected_model_name": selected["name"],
            "selected_model_type": selected["type"],
            "selected_model_params": selected["params"],
            "selection_source": "fixed",
        }
        selected_metric = selected["metrics"][metric_name]

    logger.info(
        "Selected model candidate %s with validation %s %.4f.",
        selected_metadata["selected_model_name"],
        metric_name,
        selected_metric,
    )
    return {
        "enabled": True,
        **selected_metadata,
        "selected_columns": columns,
        "selection_metric": metric_name,
        "fixed_candidates": fixed_results,
        "candidates": fixed_results,
    }
