# Course Materials Review

Last updated: 2026-06-20

This file records how the course materials inform the Green Taxi MLOps project plan.

References to `week_01`, `week_02`, and similar paths below describe the original class-material source folders only. Project notebooks should use the flat top-level notebook layout defined in `docs/notebook_structure.md`.

## Assignment

Source: `docs/source_materials/MLOps_project.pdf`

The project must simulate a real-world ML deployment process. Required deliverables are:

- Report with a maximum of 6 pages.
- Reproducible code for generating the pipeline.
- Kedro-style modular organization and orchestration preferred.
- Data unit tests and feature store component or equivalent data quality solution.
- MLflow for experimentation and model versioning.
- Main model metrics and explainability, preferably SHAP.
- Model serving and containers.
- Data drift evaluation where a production/batch sample is used.
- Tests for relevant functions and pipelines.

Important architecture implication: each component should be an independently runnable pipeline, and the full process should run sequentially.

## Software Materials

Sources:

- `class_materials/Software-20260618/pyproject.toml`
- `class_materials/Software-20260618/readme.txt`
- `class_materials/Software-20260618/uv_guide.pdf`
- `class_materials/Software-20260618/Software_Requirements_2026.pdf`

Planning decisions:

- Use `uv` and lock dependencies for reproducibility.
- Include the course stack where compatible: Kedro, `kedro-mlflow`, MLflow, Great Expectations, Hopsworks, Optuna, Evidently, Prefect, Docker-related tooling, and JupyterLab.
- Keep Hopsworks credentials in local configuration only.
- Pin versions after the environment is created and verified.

## Theoretical Classes

Sources reviewed under `class_materials/Theoretical_Classes-20260618/`.

Project implications:

- Week 1 model development emphasizes data quality, monitoring, model registry, and production monitoring.
- Week 2 development emphasizes feature stores, Great Expectations, and training/serving consistency.
- Week 3 production deployment introduces Git, tracking, MLflow, Kedro, and orchestration.
- Week 4 deployment requirements emphasize feature parity between training and serving, model registry, package store concepts, and monitoring triggers.
- Week 5 serving emphasizes Docker, container best practices, and APIs.
- Week 6 monitoring emphasizes data drift, concept drift, performance monitoring, and logged explainability.
- Week 7 governance emphasizes explainability, policies, centralized governance, and risks/mitigations.

Planning decision: the project plan should include a production discussion even if the implementation remains a proof of concept.

## Practical Week 1: Data Unit Tests and Feature Store

Sources:

- `week_01/01_Data_Unit_Tests.ipynb`
- `week_01/01_Feature_Store_Feature_Group_Creation.ipynb`
- `week_01/02_Feature_Store_Feature_View.ipynb`
- `week_01/03_Feature_Tools.ipynb`
- `week_01/week1.ipynb`
- Great Expectations project examples under `week_01/gx_project/`

Observed organization:

- Business problem.
- Loading the data.
- Split the data.
- Build expectations.
- Validate the data.
- Load to feature store.
- Data docs and automated validation.

Project implication:

- Start with data validation and profiling before training.
- Use Great Expectations where the suite is stable.
- Keep exact expectations evidence-based after profiling Green Taxi records.
- Feature store work can be implemented as Hopsworks if credentials are available, or documented as a local feature table/catalog alternative if not.
- For Notebook 1, mirror the class data-unit-test workflow by defining a data contract, validating a monthly/batch-style dataset, parsing results into a tabular artifact, and keeping observability outputs visible in the notebook. Use lightweight reusable validation helpers first; add a full Great Expectations context later if the suite stabilizes and dependency overhead is justified.

## Practical Week 2: MLflow and Optuna

Sources:

- `week_02/01_MLflow_intro.ipynb`
- `week_02/02_Optuna_MLFlow.ipynb`
- `week_02/week2_problem.ipynb`
- `week_02/feature_utils.py`

Observed organization:

- Load dataset.
- Start experiment.
- Log relevant parameters and metrics.
- Use autologging.
- Hyperparameter tuning.
- Register best model.
- Load model for predictions.

Project implication:

- Build a baseline before Optuna.
- Log data version, selected months, target definition, parameters, metrics, model artifact, and report artifacts.
- Use MLflow registry only after model comparison is meaningful.

## Practical Week 3 and Week 4: Kedro and Deployment Requirements

Sources:

- `docs/source_materials/kedro_installation.pdf`
- `week_03/bank-example/`
- `week_04/bank-example/`
- `week_04/week_04_problem.pdf`
- `week_04/docker_recommendation/`
- `week_04/my-mlops-lab/`

Observed organization:

- `conf/base/catalog.yml`
- `conf/base/parameters.yml`
- `src/<package>/pipeline_registry.py`
- `src/<package>/pipelines/<pipeline_name>/nodes.py`
- `src/<package>/pipelines/<pipeline_name>/pipeline.py`
- `tests/pipelines/`
- `data/01_raw` through `data/08_reporting`

Observed pipeline names in the bank example:

- `ingestion`
- `data_unit_tests`
- `split_data`
- `preprocess_train`
- `split_train`
- `model_selection`
- `model_train`
- `feature_selection`
- `preprocess_batch`
- `inference`
- `production_full_train_process`
- `production_full_prediction_process`

Project implication:

- Mirror this organization but use Green Taxi specific package and pipeline names.
- Put data artifacts in the Kedro catalog.
- Keep parameter choices in YAML rather than hard-coded notebooks.
- Include tests near the relevant pipeline behavior.

## Practical Week 5: Serving and Containers

Sources:

- `week_05/bank-example/`
- `week_05/spotify_recommender/`

Observed organization:

- FastAPI app with `/health` and `/ready`.
- Pydantic request model.
- Model loaded during startup/lifespan.
- Dockerfile based on slim Python image with `uv` installation.
- Kubernetes service/deployment examples.
- Optional Streamlit pages for catalog, MLflow UI, pipeline visualization, and pipeline runs.

Project implication:

- The Green Taxi service should expose health/readiness endpoints.
- Request schema must include only serving-time features, decided after profiling and feature design.
- Docker should install from pinned project dependencies.
- Kubernetes manifests are optional unless needed for the demonstration.

## Practical Week 6: Drift

Sources:

- `week_06/00_Data_Drift.ipynb`
- `week_06/01_datadrift_PSI.ipynb`
- `week_06/02_datadrift_JS.ipynb`
- `week_06/03_NannyML_example.ipynb`
- `week_06/04_evidently.ipynb`

Observed organization:

- Conceptual drift examples.
- PSI implementation.
- JS divergence implementation.
- Univariate and multivariate drift detection.
- Evidently reports.
- Saving drift experiments in MLflow.

Project implication:

- Monthly Green Taxi files are a natural batch structure.
- Drift should compare a reference period to a later batch only after train/test months are chosen.
- If natural drift is weak, artificial drift experiments may be documented separately, not mixed with factual drift conclusions.

## Practical Week 7: Explainability

Source:

- `week_07/05_Explainability.ipynb`

Observed organization:

- Permutation importance.
- Shapley values.
- Single event explanation.

Project implication:

- Save global feature importance and SHAP plots as reproducible report artifacts.
- Add local explanation examples only after deciding what a realistic served trip request looks like.

## Official NYC TLC Documentation

Sources:

- https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_green.pdf
- https://www.nyc.gov/assets/tlc/downloads/pdf/trip_record_user_guide.pdf
- https://www.nyc.gov/assets/tlc/downloads/pdf/working_parquet_format.pdf

Relevant facts:

- Trip data is published monthly and stored as Parquet.
- Green Taxi records represent street-hail livery trips.
- The page notes that 2025 onward includes `cbd_congestion_fee`.
- Green Taxi fields include vendor ID, pickup/dropoff datetime, pickup/dropoff taxi zone IDs, passenger count, trip distance, fare amounts, rate code, payment type, trip type, congestion surcharge, and CBD congestion fee.
- TLC warns that records were provided by technology providers and accuracy is not guaranteed.
- Taxi zone lookup tables should be used to map `PULocationID` and `DOLocationID`.

Project implication:

- Schema checks must account for year/month schema variation.
- Data quality checks are central to the project, not incidental.
- Data source links and selected months must be recorded in parameters or metadata.
