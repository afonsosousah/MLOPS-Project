# Project Plan

Last updated: 2026-06-19

## Goal

Build a reproducible MLOps proof of concept using NYC TLC Green Taxi trip records. The implementation should demonstrate a real deployment-oriented workflow, not just a modeling notebook.

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

Project-specific success criteria are deferred until data profiling and target selection. Do not define final metric thresholds until the modeling problem is chosen.

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

- Download or ingestion logic for selected Green Taxi Parquet files and taxi zone lookup.
- Data profiling notebook.
- Schema summary and data quality findings.
- Candidate target assessment.

Acceptance:

- Source URLs, months, row counts, and schema differences are recorded.
- Data-dependent decisions remain deferred until evidence exists.

### Sprint 2: Data Quality and Feature Readiness

Outputs:

- Great Expectations suite or equivalent validation helpers.
- Kedro `data_unit_tests` pipeline.
- Initial feature engineering candidates.
- Optional feature store proof of concept or documented local alternative.

Acceptance:

- Data quality failures halt or clearly flag bad input.
- Expectations are justified by profiling evidence.

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
- Package/version list.
- Reproducibility instructions.
- Zip/Git-ready project.

Acceptance:

- A new user can run the pipeline from documented commands and reproduce the core results.

## Initial Repository Structure

Create this structure when implementation begins:

```text
mlops-project/
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
  src/
    green_taxi_mlops/
      __init__.py
      __main__.py
      pipeline_registry.py
      pipelines/
  tests/
```

## Candidate Pipelines

These are planned architecture components, not guaranteed implementation scope. Keep or remove them based on evidence and time.

| Pipeline | Purpose | Data-dependent decisions |
|---|---|---|
| `ingestion` | Read/download selected Green Taxi Parquet files and taxi zones | Months, sample size, schema normalization |
| `data_unit_tests` | Validate schema and quality | Exact expectations and thresholds |
| `data_profiling` | Produce profiling artifacts | Which summaries matter |
| `preprocessing_train` | Clean training data | Invalid trip filters |
| `split_data` | Build train/validation/test sets | Target and split periods |
| `feature_engineering` | Create reusable features | Feature list and serving availability |
| `model_selection` | Compare candidate models | Model families and metrics |
| `model_train` | Train final selected model | Final parameters |
| `explainability` | Generate SHAP/importance artifacts | Explanation sample and feature subset |
| `preprocessing_batch` | Prepare future/batch data | Batch period and schema |
| `model_predict` | Score batch or API inputs | Output schema |
| `data_drift` | Compare reference/current periods | Drift metric and thresholds |
| `reporting` | Save report figures/tables | Final report story |

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

