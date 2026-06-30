import logging
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_COLUMNS_TO_DROP = [
    "VendorID",
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "store_and_fwd_flag",
    "payment_type",
    "total_amount",
    "source_partition",
    "trip_id",
]


class GreenTaxiPreprocessor:
    """Fit-on-train scaling/encoding, applied consistently to train, validation, and batch.

    Expects data that already has engineered features (trip_duration_min, pickup_hour,
    is_weekend, is_rush_hour, is_night, is_airport) — see data_cleaning pipeline.
    """

    def __init__(self, parameters: dict[str, Any], target_column: str) -> None:
        self.columns_to_drop = parameters.get(
            "columns_to_drop", DEFAULT_COLUMNS_TO_DROP
        )
        self.target_column = target_column
        self.numeric_features_: list[str] = []
        self.categorical_features_: list[str] = []
        self.output_columns_: list[str] = []
        self.transformer_: ColumnTransformer | None = None

    def _prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        prepared = data.drop(columns=[self.target_column], errors="ignore")
        columns_to_drop = [
            column for column in self.columns_to_drop if column in prepared.columns
        ]
        return prepared.drop(columns=columns_to_drop)

    def fit(self, data: pd.DataFrame) -> "GreenTaxiPreprocessor":
        prepared = self._prepare(data)
        self.numeric_features_ = prepared.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()
        self.categorical_features_ = prepared.select_dtypes(
            exclude=["number", "bool"]
        ).columns.tolist()

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        self.transformer_ = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, self.numeric_features_),
                ("categorical", categorical_transformer, self.categorical_features_),
            ],
            verbose_feature_names_out=False,
        )
        self.transformer_.fit(prepared)
        self.output_columns_ = self.transformer_.get_feature_names_out().tolist()
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.transformer_ is None:
            raise ValueError("GreenTaxiPreprocessor must be fitted before transform.")
        prepared = self._prepare(data)
        transformed = self.transformer_.transform(prepared)
        return pd.DataFrame(transformed, columns=self.output_columns_, index=data.index)


def preprocessing_train(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    parameters: dict[str, Any],
    target_column: str,
) -> tuple[
    pd.DataFrame, pd.DataFrame, GreenTaxiPreprocessor, list[str], dict[str, Any]
]:
    """Fit preprocessing on X_train and transform train and validation features."""
    transformer = GreenTaxiPreprocessor(parameters, target_column).fit(X_train)
    X_train_preprocessed = transformer.transform(X_train).reset_index(drop=True)
    X_val_preprocessed = transformer.transform(X_val).reset_index(drop=True)
    production_columns = X_train_preprocessed.columns.tolist()

    report = {
        "train_rows": int(len(X_train_preprocessed)),
        "validation_rows": int(len(X_val_preprocessed)),
        "numeric_features_before_encoding": transformer.numeric_features_,
        "categorical_features_before_encoding": transformer.categorical_features_,
        "production_column_count": int(len(production_columns)),
    }

    logger.info(
        "Preprocessing fitted on %d train rows and produced %d columns.",
        len(X_train_preprocessed),
        len(production_columns),
    )
    return (
        X_train_preprocessed,
        X_val_preprocessed,
        transformer,
        production_columns,
        report,
    )