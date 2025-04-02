from fastapi import FastAPI
import gdown
import pickle
import os
import uvicorn

# Initialize FastAPI
app = FastAPI()

# Google Drive File ID & Model Path
FILE_ID = "1j_wZqYeOC4yrGn0RklbTc4wOZN-3kvaV"
MODEL_PATH = "qsar_model.pkl"

# Function to download model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# Load the model
def load_model():
    download_model()
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

# Load QSAR Model
model = load_model()

# API Endpoint for Prediction
@app.post("/predict/")
def predict(data: dict):
    try:
        features = data["features"]  # Expecting list of features
        prediction = model.predict([features])
        return {"prediction": float(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

# Run the API (Only for local testing, Render will handle this automatically)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
