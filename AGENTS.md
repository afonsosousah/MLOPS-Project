# AGENTS.md

This project implements the MLOps course project using NYC TLC Green Taxi trip records.
All project work must stay inside `C:\Users\Asus\Documents\MLOPS Project` unless a task explicitly says otherwise.

## Current Project State

Before creating or changing any notebook, section, pipeline, or report content, read and update:

- `docs/project_state.md`
- `docs/project_plan.md`
- `docs/notebook_structure.md` when notebook structure changes
- `docs/course_materials_review.md` when a new course material finding changes the plan
- `AGENTS.md` itself when a durable project rule, workflow, or source contract changes

Update the docs first, then make the requested notebook/code/report change. The docs should always describe the current state of the project, not an aspirational version that no longer matches the files.

## Source Material

Use these sources as the project contract:

- Assignment copy: `docs/source_materials/MLOps_project.pdf`
- Prior dataset decision copy: `docs/source_materials/dataset_recommendations.md`
- Kedro installation tutorial copy: `docs/source_materials/kedro_installation.pdf`
- Course materials: `class_materials/`
- Official NYC TLC trip data page: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Green taxi data dictionary: https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_green.pdf
- TLC trip records user guide: https://www.nyc.gov/assets/tlc/downloads/pdf/trip_record_user_guide.pdf
- Parquet guide: https://www.nyc.gov/assets/tlc/downloads/pdf/working_parquet_format.pdf
- Karpathy-style CLAUDE.md simplicity guidance: https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md

## Non-Negotiable Constraints

- Do not hard-code findings that require profiling the data first.
- Do not preselect final target, final feature set, outlier thresholds, drift thresholds, train/validation/test months, model family, or success criteria until the data profiling/EDA work has produced evidence.
- Keep notebooks exploratory and explanatory. Production behavior belongs in Kedro pipelines, reusable Python modules, tests, and config.
- Keep the project reproducible. Any result in the report must be regenerated from source code and a documented sample of data.
- Do not commit large raw data, credentials, MLflow databases, local caches, bulky transient debug output, or private configuration.
- Keep final delivery notebook outputs visible after the notebooks have been rerun and reviewed. The course deliverable is easier to grade when plots, tables, metrics, and explanations are visible in the submitted notebooks.
- Put credentials only in `conf/local/`, `.env`, or local environment variables, and keep them out of version control.

## Beginner-First MLOps Rules

The target reader is a student learning MLOps for the first time. Optimize notebooks for understanding before reuse.

- Prefer direct, readable notebook cells over hidden helper modules.
- Avoid helper functions unless they make a repeated step clearer.
- Do not create reusable modules before the notebook concept is understandable in plain cells.
- Keep each notebook close to the corresponding practical class flow.
- Adapt the practical class material to Green Taxi; do not copy full examples blindly.
- Introduce Kedro pipelines only after the equivalent notebook workflow is clear.
- Keep markdown explanations short and tied to the next code cell.
- If a step feels advanced, add one visible example before abstracting it.

## Karpathy/CLAUDE.md Simplicity Guardrails

Use the supplied CLAUDE.md guidance as a simplicity check before changing code, notebooks, or docs.

- State assumptions before implementing.
- If multiple interpretations exist, surface them instead of silently choosing.
- Choose the minimum code that solves the current task.
- Add no speculative flexibility, configurability, or features.
- Do not abstract single-use code.
- Touch only files required by the request.
- Do not refactor adjacent code just because it could be cleaner.
- Remove only unused imports, variables, or functions created by the current change.
- Define a verifiable goal for every non-trivial change, then verify it.
- If the change starts looking large, simplify before continuing.

## Preferred Architecture

Follow the generated Kedro-style organization now located at the workspace root:

```text
MLOPS Project/
  AGENTS.md
  README.md
  pyproject.toml
  uv.lock
  conf/
    base/
      catalog.yml
      parameters.yml
      parameters_*.yml
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
  docs/
  notebooks/
    01_data_profiling_and_validation.ipynb
    02_feature_engineering_and_feature_store.ipynb
    03_experiment_tracking_and_modeling.ipynb
    04_model_serving_and_containers.ipynb
    05_monitoring_and_drift.ipynb
    06_explainability_and_report_artifacts.ipynb
  src/
    mlops_project/
      pipeline_registry.py
      pipelines/
  tests/
```

Use the generated Kedro package name `mlops_project`.

## Pipeline Guidance

The assignment expects separated pipeline orchestration, but pipelines should come after the notebook workflow is understandable. Do not create a pipeline just to satisfy a checklist.

Possible pipeline names, only when the corresponding notebook section is stable:

- `ingestion`
- `data_unit_tests`
- `data_profiling`
- `preprocessing_train`
- `split_data`
- `feature_engineering`
- `model_selection`
- `model_train`
- `explainability`
- `preprocessing_batch`
- `model_predict`
- `data_drift`
- `reporting`

Composite pipelines can be added after the individual pipelines exist:

- `production_full_train_process`
- `production_full_prediction_process`
- `monitoring_process`

Do not implement a pipeline with placeholder logic just to satisfy the list. If the data does not justify a pipeline, document the decision and keep the pipeline out.

## Notebook Rules

Notebook names and headings should mirror the practical classes where applicable. Use section and subsection headings consistently:

- Business problem
- Dataset description
- Loading the data
- Data validation
- Feature engineering
- Experiment tracking
- Model comparison
- Explainability
- Serving or batch prediction
- Drift evaluation
- Conclusions and production notes

Notebook output should remain visible for final delivery after a clean top-to-bottom run. Important plots, tables, model metrics, SHAP summaries, and drift reports should also be saved under `data/08_reporting/` or `docs/figures/` by reproducible code, so the visible outputs are not the only source of evidence.

Before creating or extending a notebook, inspect the matching practical class week:

- Notebook 1: Week 1 data profiling, data unit tests, and validation.
- Notebook 2: Week 1 feature store and feature view material.
- Notebook 3: Week 2 MLflow and Optuna material.
- Kedro pipeline extraction: Week 3 and Week 4 Kedro examples.
- Notebook 4: Week 4 and Week 5 serving/container material.
- Notebook 5: Week 6 monitoring and drift material.
- Notebook 6: Week 7 explainability plus final report needs.

Use the practical class as a guide for sequence and concepts, not as code to copy wholesale.

## Data Guidance

Known from official TLC documentation:

- Green taxi records are monthly Parquet files.
- One row represents one trip.
- Green taxi records include pickup and dropoff datetimes, pickup and dropoff location IDs, trip distance, fare components, rate type, payment type, passenger count, and trip type.
- TLC warns that the data is provided by technology providers and may contain accuracy or completeness issues.
- Since 2025, Green Taxi data can include `cbd_congestion_fee`.

Defer until profiling:

- Which months are used for training, validation, testing, and drift.
- Whether the model predicts duration, fare, long-trip risk, or another target.
- Which records are invalid versus rare but valid.
- Which columns are available and type-stable across the selected months.
- Which columns are permissible at serving time.
- Which features are used for training and SHAP.
- What drift thresholds are meaningful.

## Technology Guidance

Use the course stack unless there is a clear compatibility problem:

- `uv` for dependency management.
- Kedro for project structure and orchestration.
- Great Expectations for data unit tests, or simple asserts where a lightweight check is more maintainable.
- MLflow for experiment tracking, artifact logging, and model registry.
- Optuna for hyperparameter search only after a baseline exists.
- SHAP or permutation importance for explainability.
- Evidently, PSI, JS divergence, or NannyML for drift, chosen after the modeling task is clear.
- FastAPI and Docker for serving.
- Prefect only if orchestration beyond Kedro commands is needed for the project demonstration.

## Testing Expectations

Add tests when behavior is implemented, not after the whole project is done.

Minimum expected tests:

- Data validation helpers.
- Feature engineering functions.
- Train/test split logic, especially time-based splits.
- Model input schema preparation.
- Prediction API request validation.
- Drift metric helpers if custom PSI or JS code is used.
- Kedro pipeline smoke tests for critical pipelines.

## Reporting Expectations

The final report is limited to 6 pages. Keep report evidence focused:

- Dataset choice and objective.
- Project planning and division of work.
- EDA and modeling conclusions.
- Main metrics.
- Explainability and feature importance.
- Production implementation proposal.
- Risks and mitigations.
- Package versions.

Do not put unsupported conclusions in the report. Every claim about data quality, model performance, explainability, or drift must point to an artifact generated by the project.
