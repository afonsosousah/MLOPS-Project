# Project Plan

Last updated: 2026-06-26

## Goal

Build a reproducible MLOps proof of concept using NYC TLC Green Taxi trip records. The implementation should demonstrate a real deployment-oriented workflow, not just a modeling notebook.

The project should stay beginner-first: each notebook should teach one main MLOps topic at a pace similar to the practical classes before code is moved into reusable modules or Kedro pipelines.

## Success Criteria

Course-level success criteria:

- Reproducible pipeline with modular components.
- Evidence-based dataset choice and business objective.
- Data quality checks.
- MLflow experiment tracking and model versioning.
- Model metrics and explainability.
- Serving API and container.
- Drift evaluation.
- Tests.
- Six-page maximum report with package versions and production discussion.

Notebook 3 currently uses `tip_amount` as a regression target. Project-specific metric thresholds remain deferred until the modeling evidence is reviewed.

## Planned Sprints

### Sprint 0: Setup and Planning

Outputs:

- Project documentation.
- Project skeleton.
- Dependency plan.
- Initial Kedro project structure.

Acceptance:

- `AGENTS.md` and docs are current.
- Project can be installed in a reproducible local environment.

### Sprint 1: Data Acquisition and Profiling

Outputs:

- Local raw download of selected Green Taxi Parquet files and taxi zone lookup.
- Reproducible ingestion logic for those raw files.
- Data profiling notebook.
- Schema summary and data quality findings.
- Candidate target assessment.

Acceptance:

- Source URLs, months, row counts, and schema differences are recorded.
- Data-dependent decisions remain deferred until evidence exists.

Current acquisition status:

- Downloaded 2024-01 through 2025-12 Green Taxi monthly Parquet files under `data/01_raw/green_taxi/YYYY/`.
- Downloaded currently listed 2026 Green Taxi monthly Parquet files, 2026-01 through 2026-04, under `data/01_raw/green_taxi/2026/`.
- Downloaded `taxi_zone_lookup.csv`.
- Later 2026 files are expected to become available monthly with TLC's normal publication delay.
- Created `notebooks/01_data_profiling_and_validation.ipynb` by hand, based on the Week 1 practical class flow.
- Notebook 1 is the style reference for future notebooks: visible steps first, minimal helper code, no hidden utility modules before the concept is clear.

### Sprint 2: Data Quality and Feature Readiness

Outputs:

- Great Expectations suite or equivalent validation checks in Notebook 1.
- Kedro `data_unit_tests` pipeline only after the notebook validation workflow is clear.
- Initial feature engineering candidates.
- Optional feature store proof of concept or documented local alternative.

Acceptance:

- Data quality failures halt or clearly flag bad input.
- Expectations are justified by profiling evidence.

Current status:

- Notebook 1 has started the Great Expectations suite in the notebook itself.
- The validation execution section still needs to be completed visibly before extracting code into reusable modules.
- Exact cleaning thresholds remain deferred until the prediction target and feature set are selected.
- Notebook 2 currently prepares `tip_amount` train and validation datasets under `data/02_intermediate/` when executed.
- Kedro `ingestion` and `data_unit_tests` pipeline folders now exist. `ingestion` only loads and enriches raw data; Great Expectations validation is centralized in `data_unit_tests`. The selected `data_unit_tests` run should include ingestion first because its validation node consumes `ingested_data` rather than raw partitions directly.
- Notebook 1 should briefly point readers from the visible notebook validation logic to these Kedro pipelines, without turning the notebook into pipeline documentation.

### Sprint 3: Baseline Modeling and MLflow

Outputs:

- Time-aware split strategy, if justified by the selected target.
- Baseline model.
- MLflow experiment logging.
- Metrics and model artifacts.

Acceptance:

- Baseline run is reproducible.
- Metrics are appropriate for the selected target.
- Model comparison does not rely on notebook-only state.

Current status:

- Notebook 3 now uses `tip_amount` regression.
- Regression metrics should include errors such as MAE/RMSE and a fit metric such as R2.
- MLflow model logging should use `skops` serialization with the narrow trusted type `numpy.dtype` instead of CloudPickle.
- MLflow model logging should use `name="model"` and an explicit conda environment to avoid MLflow 3 `artifact_path` deprecation and pip version inference warnings.
- Notebook 3 should disable MLflow environment-variable recording to avoid noisy local environment variable notices in visible outputs.

### Sprint 4: Model Selection, Explainability, and Reporting Artifacts

Outputs:

- Model comparison and optional Optuna tuning.
- Model registry candidate.
- SHAP or permutation importance artifacts.
- Report-ready plots/tables.

Acceptance:

- Best model is selected using logged evidence.
- Explainability artifacts are saved by code and referenced in the report.

### Sprint 5: Serving and Containerization

Outputs:

- FastAPI service.
- Pydantic request/response schema.
- Health/readiness endpoints.
- Dockerfile.
- Example request payload.

Acceptance:

- The API uses only serving-time features.
- The container builds and can return a prediction from a valid example.

### Sprint 6: Drift and Monitoring

Outputs:

- Drift reference/current batch definition.
- PSI, JS, Evidently, or NannyML drift report.
- MLflow drift artifacts.
- Production monitoring notes.

Acceptance:

- Drift findings distinguish natural observed drift from any artificial stress test.
- Thresholds and alerts are justified or clearly labeled as placeholders.

### Sprint 7: Final Report and Packaging

Outputs:

- Six-page report.
- Final notebooks rerun with visible reviewed outputs.
- Package/version list.
- Reproducibility instructions.
- Zip/Git-ready project.

Acceptance:

- A new user can run the pipeline from documented commands and reproduce the core results.

## Initial Repository Structure

Create this structure when implementation begins:

```text
MLOPS Project/
  README.md
  pyproject.toml
  uv.lock
  conf/
    base/
    local/
  data/
    01_raw/
    02_intermediate/
    03_primary/
    04_feature/
    05_model_input/
    06_models/
    07_model_output/
    08_reporting/
  notebooks/
    01_data_profiling_and_validation.ipynb
    02_feature_engineering_and_feature_store.ipynb
    03_experiment_tracking_and_modeling.ipynb
    04_model_serving_and_containers.ipynb
    05_monitoring_and_drift.ipynb
    06_explainability_and_report_artifacts.ipynb
  src/
    mlops_project/
      __init__.py
      __main__.py
      pipeline_registry.py
      pipelines/
  tests/
```

## Pipeline Extraction Plan

Pipelines are not the starting point. First make the workflow clear in notebooks, then extract the small stable parts needed for reproducibility.

## Notebook to Practical Class Mapping

| Notebook | Practical material to inspect first |
|---|---|
| `01_data_profiling_and_validation.ipynb` | Week 1 data profiling, data unit tests, and validation |
| `02_feature_engineering_and_feature_store.ipynb` | Week 1 feature store and feature view notebooks |
| `03_experiment_tracking_and_modeling.ipynb` | Week 2 MLflow and Optuna notebooks |
| `04_model_serving_and_containers.ipynb` | Week 4/5 deployment, serving, and container examples |
| `05_monitoring_and_drift.ipynb` | Week 6 drift and monitoring notebooks |
| `06_explainability_and_report_artifacts.ipynb` | Week 7 explainability notebook and final report needs |

Kedro code extraction should use Week 3/4 examples only after the notebook workflow is understandable.

| Pipeline | Purpose | Data-dependent decisions |
|---|---|---|
| `data_unit_tests` | Move stable Notebook 1 validation checks into a reproducible pipeline | Exact expectations and thresholds |
| `feature_engineering` | Move simple, repeated Notebook 2 transformations into reusable code | Feature list and serving availability |
| `model_train` | Move the chosen baseline/model workflow after Notebook 3 proves it | Target, split periods, and metrics |
| `model_predict` | Score batch or API inputs after serving schema is clear | Output schema |
| `data_drift` | Compare reference/current periods after Notebook 5 defines them | Drift method and thresholds |

Do not add other pipelines unless a notebook has a repeated, stable workflow that needs orchestration.

## Known Risks

- Green Taxi volumes and schemas can vary by month/year.
- Columns added in 2025, such as `cbd_congestion_fee`, may require schema normalization.
- Some fields may be unavailable at serving time if they are only known after trip completion.
- TLC states that trip data accuracy is not guaranteed.
- Hopsworks may require credentials, so the feature store component may need a local fallback.
- Full monthly Parquet files can be large, so the project should support reproducible sampling.
- Python/Kedro/MLflow versions must be checked together before locking dependencies.

## Report Outline

Keep the final report concise:

1. Use case, data source, and objective.
2. Project planning and architecture.
3. Data exploration and quality findings.
4. Modeling results and metrics.
5. Explainability and drift.
6. Production implementation, risks, mitigations, package versions.

Do not write final report conclusions before the corresponding artifacts exist.
