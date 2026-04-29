from fastapi import FastAPI
import numpy as np
import json
import joblib
import os
from tensorflow.keras.models import load_model

app = FastAPI()

# -------------------------------------------------
# BASE PATH SETUP (IMPORTANT)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebooks")

# -------------------------------------------------
# LOAD MODEL + SCALERS
# -------------------------------------------------
model = load_model(os.path.join(NOTEBOOK_DIR, "lstm_model.keras"))

scaler_X = joblib.load(os.path.join(NOTEBOOK_DIR, "scaler_X.pkl"))
scaler_y = joblib.load(os.path.join(NOTEBOOK_DIR, "scaler_y.pkl"))


def load_metrics():
    with open(os.path.join(NOTEBOOK_DIR, "metrics.json"), "r") as f:
        return json.load(f)

# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------
@app.get("/")
def home():
    return {"status": "API running"}

# -------------------------------------------------
# METRICS ENDPOINT
# -------------------------------------------------
@app.get("/metrics")
def get_metrics():
    return load_metrics()

# -------------------------------------------------
# PREDICTION ENDPOINT
# -------------------------------------------------
@app.post("/predict")
def predict(data: dict):
    """
    Expected input format:
    {
        "sequence": [[...], [...], ...]  # shape: (seq_len, num_features)
    }
    """

    sequence = np.array(data["sequence"])

    # 1. Scale input
    sequence_scaled = scaler_X.transform(sequence)

    # 2. Reshape for LSTM: (1, seq_len, features)
    sequence_scaled = sequence_scaled.reshape(
        1, sequence_scaled.shape[0], sequence_scaled.shape[1]
    )

    # 3. Predict
    pred_scaled = model.predict(sequence_scaled)

    # 4. Inverse transform
    pred = scaler_y.inverse_transform(pred_scaled)

    return {
        "prediction": pred.tolist()[0]
    }
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
