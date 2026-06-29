import logging
from typing import Any, cast

import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

REPORT_COLUMNS = ["column", "test", "statistic", "p_value", "drift_detected"]


def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Detect distribution drift between reference and current data.

    Uses the two-sample KS test for numeric columns and the chi-squared
    goodness-of-fit test for categorical columns. Results are returned as
    a cleaned DataFrame with one row per column tested.
    """
    p_value_threshold = parameters.get("p_value_threshold", 0.05)
    results = []

    numeric_cols = [
        c
        for c in reference_data.select_dtypes(include="number").columns
        if c in current_data.columns
    ]
    categorical_cols = [
        c
        for c in reference_data.select_dtypes(exclude="number").columns
        if c in current_data.columns
    ]

    for col in numeric_cols:
        ref = reference_data[col].dropna()
        cur = current_data[col].dropna()
        if ref.empty or cur.empty:
            continue
        statistic_raw, p_value_raw = cast(tuple[Any, Any], stats.ks_2samp(ref, cur))
        statistic = float(statistic_raw)
        p_value = float(p_value_raw)
        results.append(
            {
                "column": col,
                "test": "ks",
                "statistic": round(statistic, 6),
                "p_value": round(p_value, 6),
                "drift_detected": bool(p_value < p_value_threshold),
            }
        )

    for col in categorical_cols:
        ref = reference_data[col].dropna()
        cur = current_data[col].dropna()
        if ref.empty or cur.empty:
            continue
        all_cats = sorted(set(ref.unique()) | set(cur.unique()), key=str)
        ref_counts = ref.value_counts()
        cur_counts = cur.value_counts()
        ref_freq = [ref_counts.get(c, 0) for c in all_cats]
        cur_freq = [cur_counts.get(c, 0) for c in all_cats]
        ref_total = sum(ref_freq)
        cur_total = sum(cur_freq)
        if ref_total == 0 or cur_total == 0:
            continue
        expected = [r * cur_total / ref_total for r in ref_freq]
        if any(e == 0 for e in expected):
            continue
        statistic_raw, p_value_raw = cast(
            tuple[Any, Any],
            stats.chisquare(cur_freq, f_exp=expected),
        )
        statistic = float(statistic_raw)
        p_value = float(p_value_raw)
        results.append(
            {
                "column": col,
                "test": "chi2",
                "statistic": round(statistic, 6),
                "p_value": round(p_value, 6),
                "drift_detected": bool(p_value < p_value_threshold),
            }
        )

    report = pd.DataFrame(results, columns=REPORT_COLUMNS)

    if not report.empty:
        n_drifted = int(report["drift_detected"].sum())
        logger.info(
            "Drift detection complete: %d/%d columns show drift (p < %.3f).",
            n_drifted,
            len(report),
            p_value_threshold,
        )

    return report
