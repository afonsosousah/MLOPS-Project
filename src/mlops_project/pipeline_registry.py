"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    discovered_pipelines = find_pipelines(raise_errors=True)
    pipelines = discovered_pipelines.copy()

    if {"ingestion", "data_unit_tests"}.issubset(discovered_pipelines):
        pipelines["data_unit_tests"] = (
            discovered_pipelines["ingestion"]
            + discovered_pipelines["data_unit_tests"]
        )

    pipelines["__default__"] = sum(discovered_pipelines.values())
    return pipelines
