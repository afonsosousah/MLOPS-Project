from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from mlops_project.pipelines.preprocessing_batch import create_pipeline
from mlops_project.pipelines.preprocessing_batch.nodes import preprocessing_batch
from mlops_project.pipelines.preprocessing_train.nodes import preprocessing_train
from tests.pipelines.preprocessing_train.test_pipeline import (
    PREPROCESSING_PARAMS,
    _features,
)


def _fitted_transformer():
    _, _, transformer, columns, _ = preprocessing_train(
        _features(["2024-06", "2024-07"], ["Queens", "Manhattan"]),
        _features(["2025-01", "2025-02"], ["Queens", "Unknown"]),
        PREPROCESSING_PARAMS,
        "tip_amount",
    )
    return transformer, columns


def test_preprocessing_batch_reuses_saved_transformer() -> None:
    transformer, columns = _fitted_transformer()
    batch = preprocessing_batch(
        _features(["2025-08", "2025-09"], ["Bronx", "Queens"]),
        transformer,
        columns,
    )

    assert batch.columns.tolist() == columns


def test_preprocessing_batch_pipeline_creates_batch_features() -> None:
    transformer, columns = _fitted_transformer()
    catalog = DataCatalog(
        {
            "test_data": MemoryDataset(
                data=_features(["2025-08", "2025-09"], ["Bronx", "Queens"])
            ),
            "preprocessing_transformer": MemoryDataset(data=transformer),
            "production_columns": MemoryDataset(data=columns),
            "X_batch_preprocessed": MemoryDataset(),
        }
    )

    SequentialRunner().run(create_pipeline(), catalog)

    assert catalog.load("X_batch_preprocessed").columns.tolist() == columns
