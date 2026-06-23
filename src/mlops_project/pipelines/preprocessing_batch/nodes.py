import logging

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from mlops_project.pipelines.preprocessing_train.nodes import clean_data, engineer_features, COLS_TO_DROP

logger = logging.getLogger(__name__)


def preprocess_batch(analysis_data: pd.DataFrame, encoder_transform: OneHotEncoder) -> pd.DataFrame:
    """
    Preprocessing for batch/analysis data using the encoder fitted on training data.
    No target column is created — this data is for inference and drift monitoring.

    Args:
        analysis_data: 2026 analysis data from split_data pipeline.
        encoder_transform: Fitted OneHotEncoder from preprocessing_train pipeline.

    Returns:
        preprocessed_batch_data: Model-ready DataFrame (no target column).
    """
    df = clean_data(analysis_data)
    df = engineer_features(df)

    # Apply saved encoder (do not refit)
    cat_cols = ["PU_borough", "DO_borough"]
    encoded = encoder_transform.transform(df[cat_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder_transform.get_feature_names_out(cat_cols),
        index=df.index,
    )
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded_df], axis=1)

    # Drop leakage and raw columns (tip_amount / total_amount held out)
    cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    # Also drop tip_amount and total_amount (not used for batch inference)
    cols_to_drop += [c for c in ["tip_amount", "total_amount"] if c in df.columns and c not in cols_to_drop]
    df = df.drop(columns=cols_to_drop)

    df = df.fillna(-1)
    logger.info("Preprocessed batch data: %d rows, %d features.", len(df), len(df.columns))
    return df
