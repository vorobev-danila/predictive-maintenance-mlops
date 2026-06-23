from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="{{ cookiecutter.project_name }} API",
    description="Predictive maintenance inference service",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    sensor_1: float = Field(..., description="Example sensor value")
    sensor_2: float = Field(..., description="Example sensor value")


class PredictionResponse(BaseModel):
    prediction: float
    status: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    prediction = payload.sensor_1 + payload.sensor_2
    return PredictionResponse(prediction=prediction, status="success")
