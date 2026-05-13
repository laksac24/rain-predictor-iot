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



# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field, field_validator
# from fastapi.middleware.cors import CORSMiddleware
# import joblib
# import numpy as np

# app = FastAPI(
#     title="Rain Prediction API",
#     description="Predicts whether it will rain based on temperature and humidity.",
#     version="1.0.0",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"]
# )
# # Load model once at startup
# model = joblib.load("model.pkl")
# latest_prediction = {}


# # --- Schemas ---

# class PredictRequest(BaseModel):
#     temperature: float = Field(..., description="Temperature in Celsius", example=22.5)
#     humidity: float = Field(..., description="Relative humidity in percent (0–100)", example=85.0)

#     @field_validator("humidity")
#     @classmethod
#     def humidity_range(cls, v):
#         if not (0 <= v <= 100):
#             raise ValueError("humidity must be between 0 and 100")
#         return v


# class PredictResponse(BaseModel):
#     # temperature: float
#     # humidity: float
#     will_rain: bool
#     probability_of_rain: float
#     # confidence: str


# class BatchPredictRequest(BaseModel):
#     samples: list[PredictRequest] = Field(..., min_items=1, max_items=100)


# class BatchPredictResponse(BaseModel):
#     results: list[PredictResponse]
#     total: int


# # --- Helpers ---

# def build_response(temp: float, hum: float) -> PredictResponse:
#     X = np.array([[temp, hum]])
#     will_rain = bool(model.predict(X)[0])
#     prob = float(model.predict_proba(X)[0][1])

#     if prob >= 0.80 or prob <= 0.20:
#         confidence = "high"
#     elif prob >= 0.60 or prob <= 0.40:
#         confidence = "medium"
#     else:
#         confidence = "low"

#     return PredictResponse(
#         will_rain=will_rain,
#         probability_of_rain=round(prob, 4)
        
#     )


# # --- Routes ---

# @app.get("/health")
# def health():
#     return {"status": "ok"}


# @app.post("/predict", response_model=PredictResponse, summary="Single prediction")
# def predict(body: PredictRequest):

#     global latest_prediction

#     result = build_response(body.temperature, body.humidity)

#     latest_prediction = {
#         "temperature": body.temperature,
#         "humidity": body.humidity,
#         "will_rain": result.will_rain,
#         "probability_of_rain": result.probability_of_rain
#     }

#     return result


# @app.post("/predict/batch", response_model=BatchPredictResponse, summary="Batch prediction")
# def predict_batch(body: BatchPredictRequest):
#     """
#     Predict rain for up to 100 (temperature, humidity) pairs in one request.
#     """
#     results = [build_response(s.temperature, s.humidity) for s in body.samples]
#     return BatchPredictResponse(results=results, total=len(results))

# @app.get("/latest")
# def latest():

#     if not latest_prediction:
#         raise HTTPException(
#             status_code=404,
#             detail="No prediction data available yet"
#         )

#     return latest_prediction




from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, EmailStr
from fastapi.middleware.cors import CORSMiddleware

import joblib
import numpy as np

import smtplib
import os

from email.mime.text import MIMEText
from dotenv import load_dotenv
load_dotenv()


# ================= FASTAPI =================

app = FastAPI(
    title="Rain Prediction API",
    description="Predicts whether it will rain based on temperature and humidity.",
    version="1.0.0",
)

# ================= CORS =================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ================= LOAD MODEL =================

model = joblib.load("model.pkl")

# ================= GLOBAL STORAGE =================

latest_prediction = {}

# ================= SCHEMAS =================


class PredictRequest(BaseModel):

    temperature: float = Field(
        ...,
        description="Temperature in Celsius",
        example=22.5
    )

    humidity: float = Field(
        ...,
        description="Relative humidity in percent (0–100)",
        example=85.0
    )

    @field_validator("humidity")
    @classmethod
    def humidity_range(cls, v):

        if not (0 <= v <= 100):
            raise ValueError("humidity must be between 0 and 100")

        return v


class PredictResponse(BaseModel):

    will_rain: bool
    probability_of_rain: float


class BatchPredictRequest(BaseModel):

    samples: list[PredictRequest] = Field(
        ...,
        min_items=1,
        max_items=100
    )


class BatchPredictResponse(BaseModel):

    results: list[PredictResponse]
    total: int


class EmailRequest(BaseModel):

    emails: list[EmailStr]


# ================= HELPERS =================


def build_response(temp: float, hum: float) -> PredictResponse:

    X = np.array([[temp, hum]])

    will_rain = bool(model.predict(X)[0])

    prob = float(model.predict_proba(X)[0][1])

    return PredictResponse(
        will_rain=will_rain,
        probability_of_rain=round(prob, 4)
    )


def send_email_alert(receiver_emails, weather_data):

    EMAIL_USER = os.getenv("EMAIL_USER")

    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

    if not EMAIL_USER or not EMAIL_PASSWORD:

        raise Exception("EMAIL_USER or EMAIL_PASSWORD missing")

    subject = "⚠ Heavy Rain Alert"

    body = f"""
Heavy Rain Alert!

Temperature: {weather_data['temperature']} °C
Humidity: {weather_data['humidity']} %

Rain Probability: {round(weather_data['probability_of_rain'] * 100, 2)} %

Please take precautions.

- AI Weather Monitoring System
"""

    try:

        # ===== CONNECT SMTP =====

        server = smtplib.SMTP("smtp.gmail.com", 587)

        server.starttls()

        server.login(EMAIL_USER, EMAIL_PASSWORD)

        # ===== SEND TO ALL USERS =====

        for email in receiver_emails:

            msg = MIMEText(body)

            msg["Subject"] = subject

            msg["From"] = EMAIL_USER

            msg["To"] = email

            server.sendmail(
                EMAIL_USER,
                email,
                msg.as_string()
            )

            print(f"Email sent to {email}")

        server.quit()

        return True

    except Exception as e:

        print("EMAIL ERROR:", e)

        return False


# ================= ROUTES =================


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Single prediction"
)
def predict(body: PredictRequest):

    global latest_prediction

    result = build_response(
        body.temperature,
        body.humidity
    )

    latest_prediction = {
        "temperature": body.temperature,
        "humidity": body.humidity,
        "will_rain": result.will_rain,
        "probability_of_rain": result.probability_of_rain
    }

    return result


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Batch prediction"
)
def predict_batch(body: BatchPredictRequest):

    results = [
        build_response(
            s.temperature,
            s.humidity
        )
        for s in body.samples
    ]

    return BatchPredictResponse(
        results=results,
        total=len(results)
    )


@app.get("/latest")
def latest():

    if not latest_prediction:

        raise HTTPException(
            status_code=404,
            detail="No prediction data available yet"
        )

    return latest_prediction


@app.post("/send-alert")
def send_alert(data: EmailRequest):

    if not latest_prediction:

        raise HTTPException(
            status_code=404,
            detail="No weather data available yet"
        )

    success = send_email_alert(
        data.emails,
        latest_prediction
    )

    if not success:

        raise HTTPException(
            status_code=500,
            detail="Failed to send emails"
        )

    return {
        "message": "Emails sent successfully",
        "total_emails": len(data.emails)
    }