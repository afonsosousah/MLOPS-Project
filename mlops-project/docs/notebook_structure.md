# Notebook Structure

Last updated: 2026-06-19

Notebook organization should mirror the practical classes. Keep notebooks under `notebooks/` and group them by course week/topic.

Production code belongs in `src/green_taxi_mlops/`. Notebooks should call reusable code instead of becoming the only implementation.

## Naming Convention

Use the course-style numbering and title casing:

```text
notebooks/
  week_01/
    00_Green_Taxi_Data_Profiling.ipynb
    01_Data_Unit_Tests.ipynb
    02_Feature_Store_Feature_View.ipynb
  week_02/
    01_MLflow_intro.ipynb
    02_Optuna_MLFlow.ipynb
  week_03/
    01_Kedro_Pipeline_Design.ipynb
  week_04/
    01_Deployment_Requirements.ipynb
  week_05/
    01_Model_Serving.ipynb
  week_06/
    00_Data_Drift.ipynb
    01_datadrift_PSI.ipynb
    04_evidently.ipynb
  week_07/
    05_Explainability.ipynb
```

Only create notebooks when there is work to do. Do not create empty notebooks just because they are listed here.

## Standard Notebook Sections

Use these headings where applicable:

```markdown
# <Notebook Title>

## Business Problem
## Dataset Description
## Loading the Data
## Initial Data Checks
## Data Validation
## Exploratory Analysis
## Feature Engineering
## Experiment Setup
## Model Training
## Model Evaluation
## Explainability
## Batch Prediction
## Drift Evaluation
## Conclusions
## Production Notes
```

Subsections should be used for concrete steps, for example:

```markdown
### Schema Overview
### Missing Values
### Invalid Trips
### Candidate Targets
### Candidate Serving Features
### Metrics Logged to MLflow
### Artifacts Saved
```

## First Notebook: Data Profiling

Recommended first notebook:

`notebooks/week_01/00_Green_Taxi_Data_Profiling.ipynb`

Purpose:

- Download or read a small selected sample of Green Taxi monthly Parquet files.
- Record source URLs and file metadata.
- Inspect schema and column types.
- Compare schemas across selected months.
- Summarize row counts, missing values, invalid-looking records, and duplicated records.
- Identify candidate target options without finalizing them.
- Identify which columns are available at serving time versus known only after trip completion.
- Produce a short list of evidence-based data quality checks for `01_Data_Unit_Tests.ipynb`.

Do not finalize thresholds or model metrics in this notebook unless the data supports them and the rationale is documented.

## Data Unit Tests Notebook

Recommended path:

`notebooks/week_01/01_Data_Unit_Tests.ipynb`

Sections:

- Business problem.
- Data source and selected months.
- Validation scope.
- Schema expectations.
- Datetime expectations.
- Location ID expectations.
- Numeric range expectations.
- Missing value expectations.
- Validation results.
- Suite export or Kedro integration notes.

Exact checks must be based on profiling results.

## MLflow and Optuna Notebooks

Recommended paths:

- `notebooks/week_02/01_MLflow_intro.ipynb`
- `notebooks/week_02/02_Optuna_MLFlow.ipynb`

Sections:

- Experiment setup.
- Baseline run.
- Logged parameters.
- Logged metrics.
- Logged artifacts.
- Model registry candidate.
- Hyperparameter tuning, only after a baseline exists.
- Comparison table.

Do not run Optuna before a baseline and data split have been justified.

## Drift Notebooks

Recommended paths:

- `notebooks/week_06/00_Data_Drift.ipynb`
- `notebooks/week_06/01_datadrift_PSI.ipynb`
- `notebooks/week_06/04_evidently.ipynb`

Sections:

- Reference period.
- Current/batch period.
- Feature list.
- PSI or JS calculations.
- Evidently report.
- MLflow logging.
- Interpretation.
- Action recommendation.

Do not label drift as production drift unless the reference and current periods are documented.

## Explainability Notebook

Recommended path:

`notebooks/week_07/05_Explainability.ipynb`

Sections:

- Model and data version.
- Global feature importance.
- SHAP summary.
- Local explanation example.
- Limitations.
- Report-ready figures.

Local explanations should use realistic request examples only after the serving schema is decided.

