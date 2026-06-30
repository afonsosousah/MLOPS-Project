"""FastAPI app for serving the production Green Taxi tip-amount regressor."""

import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path("data/06_models/production_model.pkl")
TRANSFORMER_PATH = Path("data/04_feature/preprocessing_transformer.pkl")
COLUMNS_PATH = Path("data/04_feature/best_columns.pkl")

app = FastAPI(
    title="Green Taxi Tip Predictor",
    description="Predicts the tip amount for NYC Green Taxi credit-card trips.",
    version="1.0.0",
)

_model = None
_transformer = None
_columns: list[str] = []


@app.on_event("startup")
def load_model() -> None:
    global _model, _transformer, _columns  # noqa: PLW0603
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    if TRANSFORMER_PATH.exists():
        with open(TRANSFORMER_PATH, "rb") as f:
            _transformer = pickle.load(f)
    if COLUMNS_PATH.exists():
        with open(COLUMNS_PATH, "rb") as f:
            _columns = pickle.load(f)


class TripFeatures(BaseModel):
    """Input features for a single Green Taxi trip.

    All fields are optional so the preprocessor's imputers handle any that are absent.
    Provide engineered features (trip_duration_min, pickup_hour, etc.) if available.
    """

    RatecodeID: float | None = None
    PULocationID: int | None = None
    DOLocationID: int | None = None
    PU_borough: str | None = None
    DO_borough: str | None = None
    passenger_count: float | None = None
    trip_distance: float | None = None
    fare_amount: float | None = None
    extra: float | None = None
    mta_tax: float | None = None
    tolls_amount: float | None = None
    improvement_surcharge: float | None = None
    congestion_surcharge: float | None = None
    trip_type: float | None = None
    trip_duration_min: float | None = None
    pickup_hour: int | None = None
    pickup_dayofweek: int | None = None
    pickup_month: int | None = None
    is_weekend: int | None = None
    is_rush_hour: int | None = None
    is_night: int | None = None
    is_airport: int | None = None
    source_year: int | None = None


class PredictionResponse(BaseModel):
    predicted_tip_amount: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(trip: TripFeatures) -> PredictionResponse:
    if _model is None or _transformer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the model_train pipeline first.",
        )

    df = pd.DataFrame([trip.model_dump()])

    # Apply the fitted preprocessor (impute + scale numeric, impute + OHE categorical)
    transformed = _transformer.transform(df)

    # Select only the columns the model was trained on; fill any missing with 0
    missing = [c for c in _columns if c not in transformed.columns]
    for col in missing:
        transformed[col] = 0.0

    prediction = float(_model.predict(transformed[_columns])[0])
    return PredictionResponse(predicted_tip_amount=round(prediction, 4))
