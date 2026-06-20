# Green Taxi MLOps Project

This project is a reproducible MLOps proof of concept using NYC TLC Green Taxi trip records. The active Kedro project root is `C:\Users\Asus\Documents\MLOPS Project`.

## Current Focus

Notebook 1 profiles the downloaded raw Green Taxi parquet files and produces evidence for later decisions about target selection, data quality expectations, feature engineering, modeling, serving, and drift monitoring.

## Setup

```powershell
uv sync
```

## Run Tests

```powershell
uv run pytest
```

## Execute Notebook 1

```powershell
uv run jupyter nbconvert --execute --inplace notebooks/01_Data_Profiling_and_Quality.ipynb
```

Profiling artifacts are written to `data/08_reporting/profiling/`.
