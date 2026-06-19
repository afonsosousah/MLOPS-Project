# Dataset Recommendations for the MLOps Project

## Project fit criteria

The assignment expects a reproducible MLOps proof of concept, not just a model. A good dataset should therefore support:

- Clear business objective and success metrics.
- Modular Kedro-style pipelines: ingestion, data tests, preprocessing, training, prediction, reporting, drift.
- MLflow experiment tracking and model registration.
- Feature importance or SHAP explanations with interpretable features.
- A realistic serving API input.
- A natural batch or time dimension for drift monitoring.
- A small sample that can be shipped with the project while keeping the full data reproducible from source.

## What the class materials imply

Local class materials already include examples around:

- Bank marketing classification with Kedro pipelines, MLflow artifacts, data tests, SHAP output, batch prediction, and drift reports.
- Great Expectations/data unit tests and feature-store notebooks.
- MLflow/Optuna experimentation.
- FastAPI/Docker/Kubernetes serving examples.
- Evidently/NannyML/PSI drift examples.

Local datasets found:

| Dataset | Local evidence | Fit | Main issue |
|---|---:|---|---|
| Bank marketing | 45,211 rows in `class_materials/Practical_Classes-20260618/week_01/data/bank_data.csv` and `week_02/bank-full.csv` | Very strong for the class pipeline | The example PDF says to use it as reference, not copy it |
| Home Credit | 246,009 rows in `week_01/data/application_train.csv` plus related tables | Rich credit-risk use case | Large and already used in class materials |
| IBM HR attrition | 1,470 rows | Easy binary classification | Small, common, already used |
| Spotify tracks | 114,000 rows | Good serving/recommender example | Target definition is less natural for supervised MLOps grading |
| HR analytics job change | 19,158 rows | Good binary target | No natural time batches; drift must be simulated |

## Recommended dataset: NYC TLC taxi trip records

Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Why this is the best fit:

- Official public source with monthly Parquet files and data dictionaries.
- Natural batch structure: each month can be a production batch.
- Realistic drift story: demand, traffic, zones, fares, regulations, and seasonality change over time.
- Strong MLOps story: data quality checks catch impossible durations, negative fares, missing zones, schema changes, outliers, and duplicate trips.
- Interpretable features for SHAP: pickup hour, day of week, pickup/dropoff zones, passenger count, trip distance estimate, rate code, vendor, payment type when appropriate.
- Serving is realistic: an API can estimate trip duration or classify long-delay risk from trip request attributes.
- You can ship a small sample but keep the full source reproducible from the official page.

Suggested problem:

- Predict taxi trip duration in minutes, or classify whether a trip will be longer than expected for its route/time bucket.
- Use green taxi or yellow taxi records. Green taxi is usually more manageable; yellow taxi gives more volume.
- Train on several months, validate on a later month, and use another later month as the drift/batch sample.

Suggested metrics:

- Regression: MAE, RMSE, median absolute error, and p90 absolute error.
- Classification alternative: ROC-AUC, PR-AUC, F1, recall at chosen precision, calibration.
- Drift: PSI or Evidently reports by month for pickup hour, trip distance, zone pairs, passenger count, and model prediction distribution.

Suggested pipeline:

1. `ingestion`: download/read selected monthly Parquet files and zone lookup table.
2. `data_quality`: assert schema, valid pickup/dropoff times, positive duration, valid zone IDs, non-negative numeric fields.
3. `feature_engineering`: create hour, weekday, weekend, route pair, distance buckets, holiday flags if desired.
4. `split_data`: time-based split by month.
5. `model_train`: LightGBM/XGBoost/sklearn tree model with MLflow logging.
6. `explainability`: SHAP summary and feature importance.
7. `model_predict`: score a future monthly batch.
8. `data_drift`: compare training months versus future month.
9. `serving`: FastAPI endpoint that accepts trip request features and returns duration estimate/risk.

## Strong alternatives

| Rank | Dataset | Source | Why use it | Risk |
|---:|---|---|---|---|
| 2 | UCI Online Retail II | https://archive.ics.uci.edu/dataset/502/online+retail+ii | 1,067,371 transaction rows over two years; great for customer repurchase/churn or demand forecasting; has natural time splits and business explainability | Requires target engineering |
| 3 | UCI Seoul Bike Sharing Demand | https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand | Small, clean hourly demand regression with weather, seasons, holidays, and natural drift | Only 8,760 rows, so less impressive for MLOps scale |
| 4 | OpenML Electricity | https://www.openml.org/api/v1/json/data/151 | Classic concept-drift classification dataset with 45,312 time-ordered examples | Features are normalized and less intuitive for SHAP/report storytelling |
| 5 | UCI AI4I 2020 Predictive Maintenance | https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset | Clear industrial failure prediction, easy serving story, interpretable features and failure modes | Synthetic, with limited natural drift |

## Recommendation

Use NYC TLC taxi trip records unless you want a smaller, easier dataset. It gives the strongest match to the grading criteria because it naturally supports orchestration, data quality tests, MLflow, explainability, serving, and drift in one coherent real-world use case.

If you want the simplest implementation with lower risk, choose Seoul Bike Sharing Demand. If you want a richer business/feature-store project and are comfortable engineering labels, choose Online Retail II.
