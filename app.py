from fastapi import FastAPI
from model.train import train_model
from model.predict import predict_next

app = FastAPI(title="LMS Performance Forecasting")

@app.post("/train")
def train():
    msg = train_model()
    return {"status": msg}

@app.get("/predict")
def predict():
    pred = predict_next()
    return {"predicted_cpu_usage": pred}
