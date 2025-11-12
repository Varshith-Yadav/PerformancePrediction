from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib

app = FastAPI()

model = tf.keras.models.load_model("server_load_forecaster.h5")
scaler = joblib.load("scaler.pkl")

class LoadInput(BaseModel):
    past_loads: list  

@app.post("/forecast")
def forecast_load(data: LoadInput):
    arr = np.array(data.past_loads).reshape(-1, 1)
    arr_scaled = scaler.transform(arr)
    arr_scaled = arr_scaled.reshape((1, len(arr_scaled), 1))

    prediction = model.predict(arr_scaled)
    predicted_load = scaler.inverse_transform(prediction)
    return {"predicted_load": float(predicted_load[0][0])}


