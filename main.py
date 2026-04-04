from fastapi import FastAPI
import joblib
import numpy as np
import logging

# LOGGING SETUP
logging.basicConfig(filename="logs.txt", level=logging.INFO)

# LOAD MODEL
model = joblib.load("model.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "DeepFlow Mini API running"}

@app.post("/predict")
def predict(data: dict):
    try:
        values = np.array([data["area"], data["bedrooms"]]).reshape(1, -1)
        prediction = model.predict(values)[0]

        # LOG
        logging.info(f"Input: {data}, Prediction: {prediction}")

        return {"prediction": prediction}

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return {"error": str(e)}