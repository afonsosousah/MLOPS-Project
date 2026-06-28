import logging
from typing import Any, Dict

import mlflow
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def select_model(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Run an Optuna hyperparameter search for RandomForestRegressor.

    Each trial is logged as a nested MLflow run.  Returns a dict with the
    best hyperparameters found, ready to be passed directly to model_train.
    """
    target_col = parameters["target_col"]
    n_trials = parameters.get("optuna_n_trials", 20)

    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col].astype(float)
    X_val = val_data.drop(columns=[target_col])
    y_val = val_data[target_col].astype(float)

    numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )

    mlflow.set_tracking_uri(parameters["mlflow_tracking_uri"])
    mlflow.set_experiment(parameters["mlflow_experiment_name"])

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state": 42,
            "n_jobs": -1,
        }

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(**params)),
        ])

        with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
            model.fit(X_train, y_train)
            rmse = float(root_mean_squared_error(y_val, model.predict(X_val)))
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)

        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {**study.best_params, "random_state": 42, "n_jobs": -1}

    logger.info(
        "Optuna search complete: %d trials, best RMSE=%.4f, params=%s",
        n_trials,
        study.best_value,
        best_params,
    )

    return best_params
