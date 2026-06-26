# Project State

Last updated: 2026-06-27

## Scope

The project uses NYC TLC Green Taxi trip records for the MLOps course project. The work lives in `C:\Users\Asus\Documents\MLOPS Project`.

The assignment expects a reproducible proof of concept with data validation, MLflow, explainability, serving/containers, drift evaluation, tests, and a final report of at most 6 pages.

## Current Direction

- The project should be beginner-first and practical-class-aligned.
- The current notebook style reference is `notebooks/01_data_profiling_and_validation.ipynb`.
- Future notebooks should be simple, visible, and paced like the class notebooks.
- The old over-abstracted work in `C:\Users\Asus\Desktop\MLOPS Project_old` should be used only as a cautionary reference, not as a template.
- Do not claim helper modules, reusable validation modules, or data-unit-test pipelines exist unless they are present in the current workspace.

## Materials Reviewed

- `docs/source_materials/MLOps_project.pdf`
- `docs/source_materials/dataset_recommendations.md`
- `docs/source_materials/kedro_installation.pdf`
- `class_materials/Theoretical_Classes-20260618/`
- `class_materials/Practical_Classes-20260618/`
- `class_materials/Software-20260618/`
- `notebooks/01_data_profiling_and_validation.ipynb`
- The old generated first notebook under `C:\Users\Asus\Desktop\MLOPS Project_old`
- Karpathy-style CLAUDE.md guidance: https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md

See `docs/course_materials_review.md` for the mapping from course materials to project choices.

## Decisions Made

- Dataset family: NYC TLC Green Taxi trip records.
- Project root: `C:\Users\Asus\Documents\MLOPS Project`.
- Project organization: generated Kedro-style package with `conf/`, `data/`, `notebooks/`, `src/`, `tests/`, and `docs/`.
- Notebook organization: six flat notebooks directly under `notebooks/`.
- Notebook naming: use the simpler lowercase style started by `01_data_profiling_and_validation.ipynb`.
- Notebook output policy: keep final reviewed outputs visible for delivery.
- Simplicity rule: write notebook steps directly first; extract reusable modules or pipelines only after the workflow is clear.
- Practical class rule: inspect the matching class week before adding a notebook or section, then adapt only what is needed.

## Planned Notebooks

- `notebooks/01_data_profiling_and_validation.ipynb`
- `notebooks/02_feature_engineering_and_feature_store.ipynb`
- `notebooks/03_experiment_tracking_and_modeling.ipynb`
- `notebooks/04_model_serving_and_containers.ipynb`
- `notebooks/05_monitoring_and_drift.ipynb`
- `notebooks/06_explainability_and_report_artifacts.ipynb`

## Current Notebook Work

Notebook 1 exists as `notebooks/01_data_profiling_and_validation.ipynb`.

It currently follows the Week 1 practical class direction:

- Load data and imports.
- Define simple constants.
- Read Green Taxi Parquet data.
- Split reference and analysis datasets.
- Generate a profiling report.
- Start a Great Expectations suite.
- Leave validation and feature-store follow-up visible as next steps.

Current Notebook 1 intentionally keeps the work mostly visible in notebook cells. It uses small local helper functions only where they make the Great Expectations workflow easier to read; future work should avoid adding more helpers unless they clearly improve readability.

Notebook 1 uses this validation split:

- Reference period: 2024-01 through 2025-12.
- Analysis period: 2026-01 through 2026-04.
- This split is only for Notebook 1 validation. It is not the final modeling train/validation/test split or final drift reference/current definition.
- The full YData Profiling report is saved to `data/08_reporting/green_taxi_reference_profile.html`.
- Great Expectations checks are starter notebook checks: stable schema and official code checks plus evidence-informed anomaly flags. They are not final production thresholds.
- The 2026 analysis period currently fails the starter suite on completeness checks for operational fields such as `RatecodeID`, `payment_type`, `trip_type`, `store_and_fwd_flag`, `passenger_count`, and `congestion_surcharge`; this is a profiling finding, not a final cleaning decision.
- Notebook 1 should mention that the productionized version of this boundary is split across Kedro `ingestion` and `data_unit_tests`: ingestion loads/enriches data, while data-unit tests run the Great Expectations checks.

Notebook 2 work currently appears in `notebooks/02_data_preprocessing.ipynb`. It prepares Green Taxi modeling features and saves intermediate train/analysis datasets under `data/02_intermediate/` when executed.

Notebook 3 has been started as `notebooks/03_experiment_tracking_and_modeling.ipynb`. It is aligned with Week 2 practical material and now focuses on MLflow experiment tracking, baseline regression models, model comparison, and a small Optuna tuning run for the `tip_amount` target. MLflow sklearn model artifacts should use the safer `skops` serialization format with `numpy.dtype` explicitly trusted for the current sklearn pipelines. Model logging should use MLflow 3's `name="model"` argument and an explicit conda environment so the notebook does not emit `artifact_path` deprecation or pip version inference warnings. Disable MLflow environment-variable recording in this notebook so local variable names are not written into model logging output.

## Current Kedro Pipeline Work

- The current Kedro extraction includes `ingestion`, `data_unit_tests`, `split_data`, `preprocessing_train`, and `preprocessing_batch` pipeline folders.
- `ingestion` is intentionally limited to loading monthly Green Taxi partitions, dropping fully null `ehail_fee`, and joining stable taxi-zone borough fields.
- The raw Green Taxi source remains one partitioned dataset under `data/01_raw/green_taxi/`. The ingestion pipeline filters this single source by configured year groups: `modeling_ingested_data` contains the 2024-2025 modeling period, and `drift_ingested_data` contains the 2026 holdout period for the drift section.
- Great Expectations checks are centralized in `data_unit_tests`, which validates both `modeling_ingested_data` and `drift_ingested_data`. Failed expectations are logged and saved as warning-level findings instead of blocking pipeline execution, because 2026 completeness failures are monitoring evidence rather than automatic cleaning decisions.
- The raw Green Taxi monthly files should be loaded from `data/01_raw/green_taxi/` as partitioned Parquet data, with `data/01_raw/taxi_zone_lookup.csv` cataloged separately for the zone join.
- Kedro registry output should exclude the generated starter/example pipelines (`example_data_processing`, `example_data_science`, and the starter `reporting` pipeline). They use toy Spaceflights datasets and can collide with Green Taxi outputs such as `X_train`, `X_test`, `y_train`, and `y_test`.
- `src/mlops_project/pipeline_registry.py` should explicitly register only active Green Taxi pipelines instead of auto-discovering every folder under `src/mlops_project/pipelines/`.
- `conf/base/catalog.yml` should describe the active Green Taxi datasets only: raw partitions, zone lookup, intermediate split data, model-input splits, fitted preprocessing/model artifacts, batch outputs, explainability artifacts, and validation reports.

## Current Files Present

- `AGENTS.md`
- `README.md`
- `pyproject.toml`
- `uv.lock`
- `conf/`
- `data/`
- `docs/project_state.md`
- `docs/project_plan.md`
- `docs/course_materials_review.md`
- `docs/notebook_structure.md`
- `notebooks/01_data_profiling_and_validation.ipynb`
- `src/mlops_project/`
- `tests/`

## Deferred Decisions

These remain deferred until the notebooks provide evidence:

- Final production prediction target beyond the current `tip_amount` regression notebook.
- Success metrics and thresholds.
- Exact train/validation/test split inside the 2024-2025 modeling period.
- Exact cleaning thresholds.
- Serving-time feature set.
- Baseline model and model family.
- Hyperparameter search space.
- Drift method and alert thresholds.
- Which artifacts are needed for the final report.

## Next Recommended Step

Finish Notebook 1 at the same beginner pace:

- Complete the Great Expectations validation step.
- Show validation results visibly in the notebook.
- Add a short conclusion listing what was learned and what remains deferred.

After Notebook 1 is understandable and complete, start `notebooks/02_feature_engineering_and_feature_store.ipynb` by first reviewing the Week 1 feature store practical notebooks.

## Update Protocol

When the user asks for a new notebook or section:

1. Read `AGENTS.md`.
2. Update this file with the requested change, current assumptions, and deferred decisions.
3. Update `docs/notebook_structure.md` if the notebook outline changes.
4. Update `docs/project_plan.md` if the change affects scope, ordering, or deliverables.
5. Inspect the corresponding practical class material.
6. Then create or edit the notebook/section with the simplest visible implementation.
