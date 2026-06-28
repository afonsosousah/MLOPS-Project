"""FastAPI app for serving the production Green Taxi tip-amount regressor."""

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path("data/06_models/production_model.pkl")
COLUMNS_PATH = Path("data/06_models/columns.pkl")

app = FastAPI(
    title="Green Taxi Tip Predictor",
    description="Predicts the tip amount for NYC Green Taxi credit-card trips.",
    version="1.0.0",
)

_model = None
_columns: list[str] = []


@app.on_event("startup")
def load_model() -> None:
    global _model, _columns
    if MODEL_PATH.exists() and COLUMNS_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(COLUMNS_PATH, "rb") as f:
            _columns = pickle.load(f)


class TripFeatures(BaseModel):
    """Input features for a single Green Taxi trip.

    All fields are optional so the model's internal imputers handle any that
    are absent.
    """

    RatecodeID: Optional[float] = None
    PULocationID: Optional[int] = None
    DOLocationID: Optional[int] = None
    passenger_count: Optional[float] = None
    trip_distance: Optional[float] = None
    fare_amount: Optional[float] = None
    extra: Optional[float] = None
    mta_tax: Optional[float] = None
    tolls_amount: Optional[float] = None
    improvement_surcharge: Optional[float] = None
    congestion_surcharge: Optional[float] = None
    trip_type: Optional[float] = None
    # engineered features
    trip_duration_min: Optional[float] = None
    pickup_hour: Optional[int] = None
    pickup_dayofweek: Optional[int] = None
    pickup_month: Optional[int] = None
    is_weekend: Optional[int] = None
    is_rush_hour: Optional[int] = None
    is_night: Optional[int] = None
    is_airport: Optional[int] = None
    trip_id: Optional[int] = None
    # location / borough
    PU_borough: Optional[str] = None
    DO_borough: Optional[str] = None
    source_year: Optional[int] = None


class PredictionResponse(BaseModel):
    predicted_tip_amount: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(trip: TripFeatures) -> PredictionResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the model_train pipeline first.")

    row = {k: v for k, v in trip.model_dump().items() if v is not None}
    df = pd.DataFrame([row])

    feature_cols = [c for c in _columns if c in df.columns]
    missing = [c for c in _columns if c not in df.columns]
    for col in missing:
        df[col] = None

    prediction = float(_model.predict(df[_columns])[0])

    return PredictionResponse(predicted_tip_amount=round(prediction, 4))
