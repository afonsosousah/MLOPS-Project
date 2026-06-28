import logging
from typing import Any, Dict, Tuple

import mlflow
import mlflow.sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

MLFLOW_MODEL_SERIALIZATION_FORMAT = mlflow_sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
DEFAULT_SHAP_SAMPLE_SIZE = 25


def model_train(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Tuple[Pipeline, list[str], Dict[str, float], pd.DataFrame]:
    target_col = parameters["target_col"]
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col].astype(float)
    X_val = val_data.drop(columns=[target_col])
    y_val = val_data[target_col].astype(float)

    numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    model_params = parameters["model_params"]

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(**model_params)),
        ]
    )

    mlflow.set_tracking_uri(parameters["mlflow_tracking_uri"])
    mlflow.set_experiment(parameters["mlflow_experiment_name"])

    with mlflow.start_run(run_name="kedro_random_forest_regressor", nested=True):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        metrics = {
            "mae": mean_absolute_error(y_val, y_pred),
            "rmse": root_mean_squared_error(y_val, y_pred),
            "median_ae": median_absolute_error(y_val, y_pred),
            "r2": r2_score(y_val, y_pred),
        }

        mlflow.log_params(model_params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "project": "green_taxi_mlops",
                "target": target_col,
                "task": "regression",
                "stage": "kedro_model_train",
                "model_serialization_format": MLFLOW_MODEL_SERIALIZATION_FORMAT,
            }
        )
        mlflow_sklearn.log_model(
            model,
            name="model",
            serialization_format=MLFLOW_MODEL_SERIALIZATION_FORMAT,
        )

    predictions = pd.DataFrame(
        {
            "actual_tip_amount": y_val.reset_index(drop=True),
            "predicted_tip_amount": y_pred,
            "residual": y_val.reset_index(drop=True) - y_pred,
        }
    )

    logger.info("Validation RMSE: %.4f", metrics["rmse"])

    preprocessor_fitted = model.named_steps["preprocessor"]
    rf_model = model.named_steps["regressor"]

    shap_sample_size = int(parameters.get("shap_sample_size", DEFAULT_SHAP_SAMPLE_SIZE))
    X_shap = X_train.sample(
        n=min(shap_sample_size, len(X_train)),
        random_state=model_params.get("random_state", 42),
    )
    X_shap_transformed = preprocessor_fitted.transform(X_shap)
    if hasattr(X_shap_transformed, "toarray"):
        X_shap_transformed = X_shap_transformed.toarray()
    X_shap_transformed = np.asarray(X_shap_transformed, dtype=float)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_shap_transformed)

    shap.summary_plot(shap_values, X_shap_transformed, show=False, max_display=20)

    return model, X_train.columns.tolist(), metrics, predictions

