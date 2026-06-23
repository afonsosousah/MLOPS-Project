# Notebook Structure

Last updated: 2026-06-20

Notebook organization should mirror the practical classes by topic and section flow, but not by class week or folder. This project uses a small flat set of final-delivery notebooks directly under `notebooks/`.

The current style reference is the hand-written `notebooks/01_data_profiling_and_validation.ipynb`. Future notebooks should be beginner-paced, visible, and close to the relevant practical class.

Production code belongs in `src/mlops_project/`. Notebooks should call reusable code only after the concept is understandable in the notebook. Kedro pipeline design belongs in code and project docs, not in a separate notebook.

## Output Policy

Keep notebook outputs visible for final delivery after a clean top-to-bottom run. The course deliverable is easier to review when plots, tables, validation results, MLflow summaries, SHAP figures, and drift reports are visible in the notebook files.

Visible outputs do not replace reproducibility. Important artifacts should also be saved by code under `data/08_reporting/` or `docs/figures/` when that is useful and easy to understand.

## Naming Convention

Use exactly these six notebooks:

```text
notebooks/
  01_data_profiling_and_validation.ipynb
  02_feature_engineering_and_feature_store.ipynb
  03_experiment_tracking_and_modeling.ipynb
  04_model_serving_and_containers.ipynb
  05_monitoring_and_drift.ipynb
  06_explainability_and_report_artifacts.ipynb
```

Only create notebooks when there is work to do. Do not create empty notebooks just because they are listed here.

## Simplicity Rules

- Prefer explicit notebook cells over imported helper modules.
- Use helper functions only when the same idea is repeated and the function makes the notebook easier to read.
- Do not create a new `src/` module until the notebook has made the workflow clear.
- Add only the class concept needed for the current notebook.
- Do not add optional features, extra abstractions, or configurable frameworks unless the assignment or practical class requires them.

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
## Conclusions and Production Notes
```

## Notebook 1: Data Profiling and Validation

Path: `notebooks/01_data_profiling_and_validation.ipynb`

Practical class material to inspect first:

- `class_materials/Practical_Classes-20260618/week_01/week1.ipynb`
- `class_materials/Practical_Classes-20260618/week_01/01_Data_Unit_Tests.ipynb`

Purpose:

- Load the Green Taxi data.
- Add visible source-month metadata and summarize row counts/schema variants.
- Split reference and analysis datasets by time for validation only.
- Profile the full reference data and save the report artifact.
- Keep the profiling section as the main place for detailed data-quality findings.
- Build starter Great Expectations checks based on observed data and official TLC code sets.
- Validate the analysis dataset and show a readable validation summary.
- Briefly explain that the same production responsibility is now separated into the Kedro `ingestion` and `data_unit_tests` pipelines.
- Keep outputs visible so the validation process is understandable.

## Notebook 2: Feature Engineering and Feature Store

Path: `notebooks/02_feature_engineering_and_feature_store.ipynb`

Practical class material to inspect first:

- `class_materials/Practical_Classes-20260618/week_01/01_Feature_Store_Feature_Group_Creation.ipynb`
- `class_materials/Practical_Classes-20260618/week_01/02_Feature_Store_Feature_View.ipynb`

Purpose:

- Define simple candidate serving-time features.
- Show transformations directly in the notebook first.
- Explain which columns are not available at serving time.
- Demonstrate Hopsworks if credentials are available, or document a local Kedro catalog fallback.

## Notebook 3: Experiment Tracking and Modeling

Path: `notebooks/03_experiment_tracking_and_modeling.ipynb`

Practical class material to inspect first:

- `class_materials/Practical_Classes-20260618/week_02/week2_problem.ipynb`
- `class_materials/Practical_Classes-20260618/week_02/01_MLflow_intro.ipynb`
- `class_materials/Practical_Classes-20260618/week_02/02_Optuna_MLFlow.ipynb`

Purpose:

- Start with a simple baseline model.
- Log parameters, metrics, and artifacts to MLflow.
- Compare models only after the baseline works.
- Add Optuna only if the baseline and metric choice are clear.

## Kedro Extraction: No Dedicated Notebook

Practical class material to inspect first:

- `class_materials/Practical_Classes-20260618/week_03/kedro_installation.pdf`
- `class_materials/Practical_Classes-20260618/week_03/bank-example/`
- `class_materials/Practical_Classes-20260618/week_04/bank-example/`

Kedro pipelines should be created when a notebook workflow is stable enough to move into reusable code. Do not create separate Kedro design notebooks.

## Notebook 4: Model Serving and Containers

Path: `notebooks/04_model_serving_and_containers.ipynb`

Practical class material to inspect first:

- `class_materials/Practical_Classes-20260618/week_04/docker_recommendation/`
- `class_materials/Practical_Classes-20260618/week_05/spotify_recommender/`

Purpose:

- Define a simple serving request using only serving-time features.
- Show the FastAPI behavior and example payloads.
- Document Docker build and run commands.

## Notebook 5: Monitoring and Drift

Path: `notebooks/05_monitoring_and_drift.ipynb`

Practical class material to inspect first:

- `class_materials/Practical_Classes-20260618/week_06/00_Data_Drift.ipynb`
- `class_materials/Practical_Classes-20260618/week_06/01_datadrift_PSI.ipynb`
- `class_materials/Practical_Classes-20260618/week_06/04_evidently.ipynb`

Purpose:

- Define reference and analysis periods after splits are chosen.
- Use one drift method first.
- Add a richer report only if it helps explain the result.
- Distinguish observed drift from artificial stress tests.

## Notebook 6: Explainability and Report Artifacts

Path: `notebooks/06_explainability_and_report_artifacts.ipynb`

Practical class material to inspect first:

- `class_materials/Practical_Classes-20260618/week_07/05_Explainability.ipynb`

Purpose:

- Generate simple feature importance or SHAP explanations for the selected model.
- Include local explanations only after realistic examples are defined.
- Save report-ready figures, metric tables, and package/version evidence.
