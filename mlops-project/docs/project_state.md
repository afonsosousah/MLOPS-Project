# Project State

Last updated: 2026-06-19

## Scope

The project will use NYC TLC Green Taxi trip records for the MLOps course project. The work must live in `mlops-project/`.

The assignment requires a proof of concept that is reproducible and organized as a modular pipeline. Expected components are data unit tests or feature store work, MLflow, model metrics, explainability, serving/containers, drift evaluation, tests, and a final report of at most 6 pages.

## Materials Reviewed

- `../MLOps_project.pdf`
- `../dataset_recommendations.md`
- `../class_materials/Theoretical_Classes-20260618/`
- `../class_materials/Practical_Classes-20260618/`
- `../class_materials/Software-20260618/`
- Official NYC TLC trip record page and linked Green Taxi documentation

See `docs/course_materials_review.md` for the mapping from course materials to project choices.

## Decisions Made

- Dataset family: NYC TLC Green Taxi trip records.
- Project root: `mlops-project/`.
- Preferred project organization: Kedro-style package with `conf/`, `data/`, `notebooks/`, `src/`, `tests/`, and `docs/`.
- Notebook organization: mirror the practical class topic/week structure.
- Documentation update rule: before a new notebook or section is created, update `AGENTS.md` and the relevant docs to reflect the current project state.
- Known external data format: monthly Parquet files with a taxi zone lookup table.

## Deferred Decisions

These must not be finalized until the Green Taxi data is downloaded and profiled:

- Final prediction target.
- Final success metrics and thresholds.
- Months used for train, validation, test, and drift/batch simulation.
- Exact schema contract and required/optional columns for selected months.
- Validity thresholds for duration, distance, fare, passenger count, and location IDs.
- Serving-time feature set.
- Baseline model and model family.
- Hyperparameter search space.
- Drift metrics and alert thresholds.
- Which artifacts are important enough for the 6-page report.

## Current Files Created

- `AGENTS.md`
- `docs/project_state.md`
- `docs/project_plan.md`
- `docs/course_materials_review.md`
- `docs/notebook_structure.md`

## Next Recommended Step

Create the project skeleton and environment files, then create the first profiling notebook:

- `notebooks/week_01/01_Data_Unit_Tests.ipynb`
- Or a narrower first notebook named `notebooks/week_01/00_Green_Taxi_Data_Profiling.ipynb` if the team wants profiling before formal Great Expectations suites.

Before creating either notebook, update this file with the exact notebook path and intended sections.

## Update Protocol

When the user asks for a new notebook or section:

1. Read `AGENTS.md`.
2. Update this file with the requested change, current assumptions, and any deferred decisions.
3. Update `docs/notebook_structure.md` if the notebook outline changes.
4. Update `docs/project_plan.md` if the change affects scope, ordering, or deliverables.
5. Then create or edit the notebook/section.

