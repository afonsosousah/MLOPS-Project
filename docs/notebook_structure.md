# Notebook Structure

Last updated: 2026-06-20

Notebook organization should mirror the practical classes by topic and section flow, but not by class week or folder. This project uses a small flat set of final-delivery notebooks directly under `notebooks/`.

Production code belongs in `src/mlops_project/`. Notebooks should call reusable code instead of becoming the only implementation. Kedro pipeline design belongs in code and project docs, not in a separate notebook.

## Output Policy

Keep notebook outputs visible for final delivery after a clean top-to-bottom run. The course deliverable is easier to review when plots, tables, validation results, MLflow summaries, SHAP figures, and drift reports are visible in the notebook files.

Visible outputs do not replace reproducibility. Important artifacts should also be saved by code under `data/08_reporting/` or `docs/figures/`.

## Naming Convention

Use exactly these six notebooks:

```text
notebooks/
  01_Data_Profiling_and_Quality.ipynb
  02_Feature_Engineering_and_Feature_Store.ipynb
  03_Experiment_Tracking_and_Modeling.ipynb
  04_Model_Serving_and_Containers.ipynb
  05_Monitoring_and_Drift.ipynb
  06_Explainability_and_Report_Artifacts.ipynb
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

## Notebook 1: Data Profiling and Quality

Path: `notebooks/01_Data_Profiling_and_Quality.ipynb`

Purpose:

- Record Green Taxi source URLs and file metadata.
- Download or read a small selected sample of monthly Parquet files.
- Inspect schema and column types.
- Compare schemas across selected months.
- Summarize row counts, missing values, invalid-looking records, and duplicated records.
- Identify candidate target options without finalizing them prematurely.
- Identify which columns are available at serving time versus known only after trip completion.
- Build evidence-based data quality checks with Great Expectations or equivalent validation helpers.
- Show the first reusable data unit test results generated from project validation helpers and save them as reporting artifacts.

Do not finalize thresholds or model metrics in this notebook unless the data supports them and the rationale is documented.

## Notebook 2: Feature Engineering and Feature Store

Path: `notebooks/02_Feature_Engineering_and_Feature_Store.ipynb`

Purpose:

- Define candidate serving-time features.
- Develop feature transformations in a way that can move into reusable pipeline code.
- Document train-time-only columns that must not leak into serving.
- Demonstrate Hopsworks feature store usage if credentials are available, or document a local Kedro catalog fallback.
- Save feature metadata needed for training, serving, and report interpretation.

Feature choices must remain tied to profiling evidence and the selected target.

## Notebook 3: Experiment Tracking and Modeling

Path: `notebooks/03_Experiment_Tracking_and_Modeling.ipynb`

Purpose:

- Define split strategy after target selection.
- Train a baseline model before tuning.
- Log parameters, metrics, artifacts, data version, and selected months to MLflow.
- Compare candidate models.
- Run Optuna only after the baseline and metric choice are justified.
- Identify a model registry candidate when evidence supports it.

Model comparison should rely on reproducible code and logged artifacts, not notebook-only state.

## Notebook 4: Model Serving and Containers

Path: `notebooks/04_Model_Serving_and_Containers.ipynb`

Purpose:

- Define the serving schema from serving-time features only.
- Document FastAPI request and response behavior.
- Include valid example payloads and expected response shape.
- Capture health and readiness endpoint behavior.
- Document Docker build/run commands and any container constraints.

The serving schema must not include fields that are only known after trip completion.

## Notebook 5: Monitoring and Drift

Path: `notebooks/05_Monitoring_and_Drift.ipynb`

Purpose:

- Define reference and current/batch periods after data splits are chosen.
- Select PSI, JS divergence, Evidently, or NannyML based on the modeling task and available features.
- Generate drift artifacts and visible interpretation.
- Log drift outputs to MLflow when useful.
- Distinguish natural observed drift from any artificial stress test.

Do not label drift as production drift unless the reference and current periods are documented.

## Notebook 6: Explainability and Report Artifacts

Path: `notebooks/06_Explainability_and_Report_Artifacts.ipynb`

Purpose:

- Generate SHAP or permutation importance artifacts for the selected model.
- Include local explanation examples only after realistic request examples are defined.
- Save report-ready figures, metric tables, and package/version evidence.
- Record limitations and production risk/mitigation evidence for the final report.

Every report-ready conclusion should point to a reproducible artifact or visible notebook output.
