import requests
import joblib
import os
from fastapi import FastAPI
import numpy as np

app = FastAPI()

# Google Drive File ID
GDRIVE_FILE_ID = "1j_wZqYeOC4yrGn0RklbTc4wOZN-3kvaV"
MODEL_PATH = "qsar_model.pkl"

def download_model():
    """Downloads model from Google Drive if not present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading QSAR model...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        response = requests.get(url)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Download complete.")

# Download model at startup
download_model()

# Load model
model = joblib.load(MODEL_PATH)

@app.get("/")
def read_root():
    return {"message": "QSAR API is running"}

@app.post("/predict/")
def predict(features: list):
    """Takes a list of features and returns the predicted pIC50 value."""
    try:
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        return {"pIC50": prediction}
    except Exception as e:
        return {"error": str(e)}

