# Project State

Last updated: 2026-06-20

## Scope

The project will use NYC TLC Green Taxi trip records for the MLOps course project. The work must live in `C:\Users\Asus\Documents\MLOPS Project`.

The assignment requires a proof of concept that is reproducible and organized as a modular pipeline. Expected components are data unit tests or feature store work, MLflow, model metrics, explainability, serving/containers, drift evaluation, tests, and a final report of at most 6 pages.

## Materials Reviewed

- `docs/source_materials/MLOps_project.pdf`
- `docs/source_materials/dataset_recommendations.md`
- `docs/source_materials/kedro_installation.pdf`
- `class_materials/Theoretical_Classes-20260618/`
- `class_materials/Practical_Classes-20260618/`
- `class_materials/Software-20260618/`
- Official NYC TLC trip record page and linked Green Taxi documentation

See `docs/course_materials_review.md` for the mapping from course materials to project choices.

## Decisions Made

- Dataset family: NYC TLC Green Taxi trip records.
- Project root: `C:\Users\Asus\Documents\MLOPS Project`.
- Preferred project organization: Kedro-style package with `conf/`, `data/`, `notebooks/`, `src/`, `tests/`, and `docs/`.
- Notebook organization: mirror the practical class topic sequence, but use six flat notebooks directly under `notebooks/` rather than class week folders or project-topic subfolders.
- Notebook output policy: keep final reviewed outputs visible for delivery, while also saving important artifacts through reproducible code.
- Documentation update rule: before a new notebook or section is created, update `AGENTS.md` and the relevant docs to reflect the current project state.
- Known external data format: monthly Parquet files with a taxi zone lookup table.
- Local raw data location: Green Taxi monthly Parquet files live under year folders in `data/01_raw/green_taxi/YYYY/`; the taxi zone lookup lives under `data/01_raw/metadata/`.
- Local raw data has been copied into the generated Kedro `data/01_raw/` structure for reproducible local work. The generated Kedro `.gitignore` keeps `data/**` out of version control except folder placeholders and example data.
- Downloaded raw coverage as of 2026-06-19: all listed Green Taxi monthly Parquet files for 2024 and 2025, plus 2026 January through April. Later 2026 files were not listed on the official TLC page yet.

## Planned Notebooks

- `notebooks/01_Data_Profiling_and_Quality.ipynb`
- `notebooks/02_Feature_Engineering_and_Feature_Store.ipynb`
- `notebooks/03_Experiment_Tracking_and_Modeling.ipynb`
- `notebooks/04_Model_Serving_and_Containers.ipynb`
- `notebooks/05_Monitoring_and_Drift.ipynb`
- `notebooks/06_Explainability_and_Report_Artifacts.ipynb`

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

## Current Notebook Work

Notebook 1 exists as `notebooks/01_Data_Profiling_and_Quality.ipynb` and is being extended with evidence-based data unit tests.

Intended sections:

- Business Problem
- Dataset Description
- Loading the Data
- Initial Data Checks
- Data Validation
- Data Unit Tests
- Exploratory Analysis
- Candidate Targets
- Serving-Time Features
- Conclusions and Production Notes

Implementation assumptions for Notebook 1:

- Use the already downloaded Green Taxi parquet files under `data/01_raw/green_taxi/`.
- Use `data/01_raw/metadata/taxi_zone_lookup.csv` only for location metadata checks, not for feature decisions yet.
- Add reusable profiling helpers under `src/mlops_project/`.
- Add reusable validation helpers and a Kedro `data_unit_tests` pipeline under `src/mlops_project/pipelines/data_unit_tests/`.
- Save reproducible profiling artifacts under `data/08_reporting/profiling/`.
- Save data unit test results under `data/08_reporting/profiling/data_unit_test_results.csv`.
- Keep the final target, split months, model family, metric thresholds, outlier thresholds, serving schema, and drift strategy deferred.

## Current Files Created

- `AGENTS.md`
- `.gitignore`
- `data/01_raw/README.md`
- `data/01_raw/green_taxi/2024/`
- `data/01_raw/green_taxi/2025/`
- `data/01_raw/green_taxi/2026/`
- `data/01_raw/metadata/taxi_zone_lookup.csv`
- `data/08_reporting/profiling/`
- `docs/project_state.md`
- `docs/project_plan.md`
- `docs/course_materials_review.md`
- `docs/notebook_structure.md`
- `README.md`
- `pyproject.toml`
- `uv.lock`
- `notebooks/01_Data_Profiling_and_Quality.ipynb`
- `src/mlops_project/__init__.py`
- `src/mlops_project/data_validation.py`
- `src/mlops_project/profiling.py`
- `src/mlops_project/pipelines/data_unit_tests/`
- `tests/test_data_validation.py`
- `tests/test_profiling.py`

## Current Data Unit Test Implementation

The first formal data unit tests have been added to Notebook 1 and reusable code:

- Required core columns exist for Green Taxi profiling.
- Pickup and dropoff datetimes are present and parseable.
- Trip duration is positive for valid candidate duration modeling rows.
- Pickup and dropoff location IDs are present.
- Distance and fare values are profiled for invalid-looking negative values.
- Schema evolution is handled without requiring `cbd_congestion_fee` in pre-2025 files.

These tests clearly distinguish enforceable checks from warnings or deferred policy decisions.

The reusable implementation lives in `src/mlops_project/data_validation.py`.
The Kedro pipeline lives in `src/mlops_project/pipelines/data_unit_tests/`.
The pipeline output is `data/08_reporting/profiling/data_unit_test_results.csv`.

## Next Recommended Step

Review Notebook 1 profiling and validation findings to define the feature engineering scope for:

- `notebooks/02_Feature_Engineering_and_Feature_Store.ipynb`

Notebook 1 now records row counts, schema differences across 2024-2026, nullable columns, invalid-looking trip records, duplicate records, temporal coverage, candidate target feasibility, and the effect of the 2025+ `cbd_congestion_fee` column. The next step is to narrow candidate serving-time features without finalizing train/test split, model family, or drift thresholds.

Before creating this notebook, update this file with the exact intended sections.

## Update Protocol

When the user asks for a new notebook or section:

1. Read `AGENTS.md`.
2. Update this file with the requested change, current assumptions, and any deferred decisions.
3. Update `docs/notebook_structure.md` if the notebook outline changes.
4. Update `docs/project_plan.md` if the change affects scope, ordering, or deliverables.
5. Then create or edit the notebook/section.
