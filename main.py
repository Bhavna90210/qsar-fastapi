import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import io

app = FastAPI()

# Load model from Google Drive
GDRIVE_URL = "https://drive.google.com/uc?id=1j_wZqYeOC4yrGn0RklbTc4wOZN-3kvaV"

response = requests.get(GDRIVE_URL)
model = pickle.load(io.BytesIO(response.content))

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])
    return {"pIC50": prediction[0]}
