import pandas as pd
import logging

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

COLS_TO_DROP = [
    "total_amount",          # leakage
    "payment_type",          # constant after filtering
    "lpep_dropoff_datetime",
    "store_and_fwd_flag",
    "VendorID",
]

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove features not used for modelling."""

    cols_to_drop = [
        c for c in COLS_TO_DROP
        if c in df.columns
    ]

    logger.info(
    "Feature selection completed: removed %d columns, %d remaining.",
    len(cols_to_drop),
    len(df.columns) - len(cols_to_drop),
)
    return df.drop(columns=cols_to_drop)
