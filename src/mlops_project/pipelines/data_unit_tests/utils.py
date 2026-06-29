from typing import Any

import great_expectations as gx
import pandas as pd


def _add_if_present(
    suite: gx.ExpectationSuite,
    df: pd.DataFrame,
    column: str,
    expectation: Any,
) -> None:
    if column in df.columns:
        suite.add_expectation(expectation)


def _parse_results(results: Any) -> pd.DataFrame:
    rows = []
    for result in results.results:
        kwargs = result.expectation_config.kwargs or {}
        column = kwargs.get("column", "")
        if not column and kwargs.get("column_A") and kwargs.get("column_B"):
            column = f"{kwargs['column_A']} > {kwargs['column_B']}"
        rows.append(
            {
                "success": result.success,
                "severity": "pass" if result.success else "warning",
                "expectation_type": result.expectation_config.type,
                "column": column,
                "min_value": kwargs.get("min_value", ""),
                "max_value": kwargs.get("max_value", ""),
                "value_set": str(kwargs.get("value_set", "")),
                "unexpected_count": result.result.get("unexpected_count", ""),
                "unexpected_percent": result.result.get("unexpected_percent", ""),
                "observed_value": str(result.result.get("observed_value", "")),
            }
        )
    return pd.DataFrame(rows)
