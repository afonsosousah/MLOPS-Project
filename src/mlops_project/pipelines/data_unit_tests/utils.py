import pandas as pd


def _parse_results(results) -> pd.DataFrame:
    rows = []
    for r in results.results:
        kwargs = r.expectation_config.kwargs or {}
        column = kwargs.get("column", "")
        if not column and kwargs.get("column_A") and kwargs.get("column_B"):
            column = f"{kwargs['column_A']} > {kwargs['column_B']}"
        rows.append(
            {
                "success": r.success,
                "expectation_type": r.expectation_config.type,
                "column": column,
                "min_value": kwargs.get("min_value", ""),
                "max_value": kwargs.get("max_value", ""),
                "value_set": str(kwargs.get("value_set", "")),
                "unexpected_count": r.result.get("unexpected_count", ""),
                "unexpected_percent": r.result.get("unexpected_percent", ""),
                "observed_value": str(r.result.get("observed_value", "")),
            }
        )
    return pd.DataFrame(rows)
