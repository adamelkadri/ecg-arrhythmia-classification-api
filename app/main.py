"""
FastAPI service for ECG heartbeat classification.

Run locally from the repo root:

    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.

Endpoints:
    GET  /health   -> liveness + whether the model is loaded
    POST /predict  -> classify a single ECG heartbeat
"""
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src import config
from src.predict import load_model, predict_signal


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model on startup so the first request isn't slow. Allow the API
    # to boot even without weights; /health reports it and /predict returns 503.
    try:
        load_model()
    except FileNotFoundError:
        pass
    yield


app = FastAPI(
    title="ECG Arrhythmia Classification API",
    description=(
        "Classify a single ECG heartbeat into one of five MIT-BIH arrhythmia "
        "classes (N, S, V, F, Q) using a 1D CNN trained in PyTorch."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Request / response schemas (Pydantic v2)
# --------------------------------------------------------------------------- #
class ECGRequest(BaseModel):
    signal: List[float] = Field(
        ...,
        description=(
            f"ECG heartbeat as a list of floats. Ideally {config.SIGNAL_LENGTH} "
            "samples; shorter inputs are zero-padded and longer ones truncated."
        ),
        examples=[[0.0] * config.SIGNAL_LENGTH],
    )

    @field_validator("signal")
    @classmethod
    def signal_not_empty(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("signal must contain at least one value")
        return v


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_class_name: str
    predicted_class_description: str
    confidence: float
    class_probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check + whether the trained model is available."""
    return HealthResponse(status="ok", model_loaded=config.MODEL_PATH.exists())


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ECGRequest) -> PredictionResponse:
    """Classify one ECG heartbeat and return class + probabilities."""
    try:
        result = predict_signal(request.signal)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python -m src.train`.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return PredictionResponse(**result)


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "ECG Arrhythmia Classification API. See /docs."}
