# from fastapi import FastAPI
# from pydantic import BaseModel

# app = FastAPI()


# class SensorData(BaseModel):
#     temperature: float
#     humidity: float


# @app.get("/")
# def home():
#     return {"message": "FastAPI server is running"}


# @app.post("/sensor")
# def receive_sensor_data(data: SensorData):
#     print("Received Sensor Data:")
#     print(f"Temperature: {data.temperature}")
#     print(f"Humidity: {data.humidity}")

#     return {
#         "rain_prediction": "72%"
#     }



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI(
    title="Rain Prediction API",
    description="Predicts whether it will rain based on temperature and humidity.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
# Load model once at startup
model = joblib.load("model.pkl")
latest_prediction = {}


# --- Schemas ---

class PredictRequest(BaseModel):
    temperature: float = Field(..., description="Temperature in Celsius", example=22.5)
    humidity: float = Field(..., description="Relative humidity in percent (0–100)", example=85.0)

    @field_validator("humidity")
    @classmethod
    def humidity_range(cls, v):
        if not (0 <= v <= 100):
            raise ValueError("humidity must be between 0 and 100")
        return v


class PredictResponse(BaseModel):
    # temperature: float
    # humidity: float
    will_rain: bool
    probability_of_rain: float
    # confidence: str


class BatchPredictRequest(BaseModel):
    samples: list[PredictRequest] = Field(..., min_items=1, max_items=100)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total: int


# --- Helpers ---

def build_response(temp: float, hum: float) -> PredictResponse:
    X = np.array([[temp, hum]])
    will_rain = bool(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    if prob >= 0.80 or prob <= 0.20:
        confidence = "high"
    elif prob >= 0.60 or prob <= 0.40:
        confidence = "medium"
    else:
        confidence = "low"

    return PredictResponse(
        will_rain=will_rain,
        probability_of_rain=round(prob, 4)
        
    )


# --- Routes ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, summary="Single prediction")
def predict(body: PredictRequest):

    global latest_prediction

    result = build_response(body.temperature, body.humidity)

    latest_prediction = {
        "temperature": body.temperature,
        "humidity": body.humidity,
        "will_rain": result.will_rain,
        "probability_of_rain": result.probability_of_rain
    }

    return result


@app.post("/predict/batch", response_model=BatchPredictResponse, summary="Batch prediction")
def predict_batch(body: BatchPredictRequest):
    """
    Predict rain for up to 100 (temperature, humidity) pairs in one request.
    """
    results = [build_response(s.temperature, s.humidity) for s in body.samples]
    return BatchPredictResponse(results=results, total=len(results))

@app.get("/latest")
def latest():

    if not latest_prediction:
        raise HTTPException(
            status_code=404,
            detail="No prediction data available yet"
        )

    return latest_prediction