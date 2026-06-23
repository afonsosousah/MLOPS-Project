# Project State

Last updated: 2026-06-23

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
- Notebook writing style: concise markdown, natural use of "we", direct formal wording for factual sections, and findings placed near the profiling or validation output that supports them.

## Planned Notebooks

- `notebooks/01_data_profiling_and_validation.ipynb`
- `notebooks/02_feature_engineering_and_feature_store.ipynb`
- `notebooks/03_experiment_tracking_and_modeling.ipynb`
- `notebooks/04_model_serving_and_containers.ipynb`
- `notebooks/05_monitoring_and_drift.ipynb`
- `notebooks/06_explainability_and_report_artifacts.ipynb`

## Current Notebook Work

Notebook 1 exists as `notebooks/01_data_profiling_and_validation.ipynb`.

It follows the Week 1 practical class direction:

- Load data and imports.
- Define simple constants and paths.
- Read Green Taxi monthly Parquet data with a visible `source_month`.
- Print compact loading diagnostics and split reference/analysis datasets by month.
- Generate and save a full profiling report as the main place for data-quality findings.
- Do not keep a separate initial-checks section when the same content is covered by profiling.
- Build a starter Great Expectations suite.
- Validate the later analysis months and summarize results visibly.
- Leave target, feature, cleaning-threshold, modeling-split, and drift-threshold decisions deferred.
- Use a single `#` notebook title and numbered `##` sections.
- Use a first-cell `Sections` index and concise section descriptions.

Current Notebook 1 intentionally keeps the work mostly visible in notebook cells. It uses small local helper functions only where they make the Great Expectations workflow easier to read; future work should avoid adding more helpers unless they clearly improve readability.

Notebook 1 uses this validation split:

- Reference period: 2024-01 through 2025-12.
- Analysis period: 2026-01 through 2026-04.
- This split is only for Notebook 1 validation. It is not the final modeling train/validation/test split or final drift reference/current definition.
- The full YData Profiling report is saved to `data/08_reporting/green_taxi_reference_profile.html`.
- Great Expectations checks are starter notebook checks: stable schema and official code checks plus evidence-informed anomaly flags. They are not final production thresholds.
- The 2026 analysis period currently fails the starter suite on completeness checks for operational fields such as `RatecodeID`, `payment_type`, `trip_type`, `store_and_fwd_flag`, `passenger_count`, and `congestion_surcharge`; this is a profiling finding, not a final cleaning decision.

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

- Final prediction target.
- Success metrics and thresholds.
- Train, validation, test, and drift periods.
- Exact cleaning thresholds.
- Serving-time feature set.
- Baseline model and model family.
- Hyperparameter search space.
- Drift method and alert thresholds.
- Which artifacts are needed for the final report.

## Next Recommended Step

After Notebook 1 is rerun and reviewed, start `notebooks/02_feature_engineering_and_feature_store.ipynb` by first reviewing the Week 1 feature store practical notebooks. Use Notebook 1 findings only as evidence for candidate checks and feature ideas, not as final modeling decisions.

## Update Protocol

When the user asks for a new notebook or section:

1. Read `AGENTS.md`.
2. Update this file with the requested change, current assumptions, and deferred decisions.
3. Update `docs/notebook_structure.md` if the notebook outline changes.
4. Update `docs/project_plan.md` if the change affects scope, ordering, or deliverables.
5. Inspect the corresponding practical class material.
6. Then create or edit the notebook/section with the simplest visible implementation.
