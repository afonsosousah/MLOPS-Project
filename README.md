# NYC TLC Green Taxi MLOps Project

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

This repository contains a course MLOps proof of concept for NYC TLC Green Taxi trip records. The current task is a `tip_amount` regression workflow using 2024-2025 Green Taxi data for modeling and available 2026 Green Taxi data as the drift holdout.

## Project Scope

- Ingest partitioned Green Taxi Parquet files from `data/01_raw/green_taxi/`.
- Profile and validate the data, keeping current data-unit-test failures as warning-level report findings unless explicitly promoted to blocking gates.
- Clean records, define the target, and build serving-aware features.
- Track model selection, Optuna trials, metrics, artifacts, and model lineage with MLflow.
- Train and compare sklearn regressors, keeping `dummy` as a non-eligible baseline.
- Generate explainability artifacts through native feature importance and SHAP where supported.
- Run NannyML-based drift monitoring against the 2026 holdout.
- Serve the trained model through FastAPI and Docker.

## Repository Layout

```text
conf/                 Kedro catalog, parameters, logging, and local config
data/01_raw/          Raw TLC files and taxi zone lookup
data/02_intermediate/ Ingested and cleaned staging data
data/03_primary/      Chronological train/test and train/validation splits
data/04_feature/      Fitted preprocessing and feature-selection artifacts
data/05_model_input/  Preprocessed model matrices
data/06_models/       Trained model artifacts
data/07_model_output/ Batch and validation predictions
data/08_reporting/    Metrics, validation reports, drift summaries, and plots
notebooks/            Exploration notebooks
src/mlops_project/    Kedro pipelines and FastAPI
tests/                Project tests only
```

## Environment Setup

Use `uv` from the project root:

```powershell
uv sync
```

Run project commands through the managed environment:

```powershell
uv run pytest tests -q
uv run kedro registry list
```

## Data Expectations

Raw Green Taxi files should remain in one partitioned source:

```text
data/01_raw/green_taxi/
```

The current workspace has 2024-01 through 2025-12 for modeling and 2026-01 through 2026-04 for drift monitoring. The zone lookup file is expected at:

```text
data/01_raw/taxi_zone_lookup.csv
```

## Main Kedro Pipelines

List available pipelines:

```powershell
uv run kedro registry list
```

Useful runs:

```powershell
uv run kedro run --pipeline data_quality
uv run kedro run --pipeline production_full_train_process
uv run kedro run --pipeline production_full_prediction_process
uv run kedro run --pipeline data_drift
```

The active registry includes base pipelines such as `ingestion`, `data_cleaning`, `split_data`, `split_train`, `preprocessing_train`, `feature_selection`, `model_selection`, `model_train`, `preprocessing_batch`, `model_predict`, `feature_store`, and `data_drift`, plus composite workflows such as `data_quality`, `drift_monitoring`, `production_full_train_process`, and `production_full_prediction_process`.

## Notebooks

Current notebooks:

- `notebooks/01_data_ingestion_profiling_and_validation.ipynb`
- `notebooks/02_data_preprocessing.ipynb`
- `notebooks/03_experiment_tracking_and_modeling.ipynb`

Use Kedro-aware notebooks when useful:

```powershell
uv run kedro jupyter lab
```

Keep final notebook outputs visible after reviewed reruns so tables, plots, validation results, metrics, and explanations are available for grading.

## MLflow

The shared MLflow configuration is in `conf/base/parameters.yml`:

- Tracking URI: `sqlite:///mlflow.db`
- Experiment: `green_taxi_tip_amount`
- Registered model name: `green_taxi_regressor`

Local MLflow files and databases are generated artifacts and should not be committed.

## Hopsworks Feature Store

The feature store is disabled by default in `conf/base/parameters_feature_store.yml`. The local Kedro catalog path is the default workflow.

To enable Hopsworks locally, keep credentials outside version control:

```powershell
$env:FS_API_KEY = "your_hopsworks_api_key"
$env:FS_PROJECT_NAME = "your_project_name"
```

Then set both `feature_store.enabled` and `feature_store.use_feature_store` to `true` in local or base configuration as appropriate.

## Serve The Model

Run training first so the API can load these artifacts:

- `data/06_models/production_model.pkl`
- `data/04_feature/preprocessing_transformer.pkl`
- `data/04_feature/best_columns.pkl`

Run locally:

```powershell
uv run uvicorn src.mlops_project.api.main:app --host 0.0.0.0 --port 8000
```

Or build and run the Docker image:

```powershell
docker build -t green-taxi-api .
docker run --rm -p 8000:8000 -v "${PWD}\data:/app/data" green-taxi-api
```

Endpoints:

- `GET http://localhost:8000/health`
- `POST http://localhost:8000/predict`
- `GET http://localhost:8000/docs`

Example prediction request:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/predict" `
  -ContentType "application/json" `
  -Body '{"PULocationID":74,"DOLocationID":42,"PU_borough":"Manhattan","DO_borough":"Brooklyn","passenger_count":1,"trip_distance":2.1,"fare_amount":12.0,"pickup_hour":18,"pickup_dayofweek":4,"pickup_month":1,"source_year":2026}'
```

## Verification

For README-only edits, the narrow verification is:

```powershell
uv run kedro registry list
```

For code or pipeline changes, run the narrowest relevant tests first, then the full project test suite when appropriate:

```powershell
uv run pytest tests -q
```
