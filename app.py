from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from model.train import train_model
from model.predict import (
    predict_all_metrics, 
    analyze_scaling_needs, 
    assess_crash_risk,
    load_models,
    predict_next
)
from datetime import datetime, timedelta

app = FastAPI(
    title="LMS Performance Predictor API",
    description="API for predicting Learning Management System performance metrics and scaling needs",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
@app.on_event("startup")
async def startup_event():
    try:
        load_models()
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not load models: {e}")
        print("Please train models first using /train endpoint")

# Request/Response Models
class PredictionResponse(BaseModel):
    predictions: Dict[str, List[float]]
    timestamps: List[str]
    scaling_alerts: List[Dict]
    crash_risk: Dict

class ScalingAnalysisResponse(BaseModel):
    scaling_alerts: List[Dict]
    recommendation: str

class CrashRiskResponse(BaseModel):
    risk_score: float
    risk_level: str
    risk_factors: List[str]
    recommendation: str

# Endpoints
@app.get("/")
async def root():
    return {
        "message": "LMS Performance Predictor API",
        "endpoints": {
            "/train": "Train all LSTM models",
            "/predict": "Get predictions for all metrics (next 24 hours)",
            "/predict/{hours}": "Get predictions for next N hours",
            "/scaling": "Analyze scaling needs",
            "/crash-risk": "Assess crash risk during exam periods",
            "/metrics/{metric}": "Get prediction for specific metric"
        }
    }

@app.post("/train")
async def train():
    """Train all LSTM models for LMS performance prediction"""
    try:
        msg = train_model()
        return {"status": "success", "message": msg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict", response_model=PredictionResponse)
async def predict(hours: int = 24):
    """
    Predict all performance metrics for the next N hours
    
    Returns:
    - predictions: Dictionary with predictions for each metric
    - timestamps: List of future timestamps
    - scaling_alerts: Alerts for metrics that need scaling
    - crash_risk: Risk assessment for system crashes
    """
    try:
        predictions = predict_all_metrics(hours=hours)
        
        # Generate timestamps
        now = datetime.now()
        timestamps = [(now + timedelta(hours=i)).isoformat() for i in range(1, hours + 1)]
        
        # Analyze scaling needs
        scaling_alerts = analyze_scaling_needs(predictions)
        
        # Assess crash risk
        crash_risk = assess_crash_risk(predictions)
        
        return PredictionResponse(
            predictions=predictions,
            timestamps=timestamps,
            scaling_alerts=scaling_alerts,
            crash_risk=crash_risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scaling", response_model=ScalingAnalysisResponse)
async def scaling_analysis(hours: int = 24):
    """
    Analyze when system will need scaling
    
    Returns alerts for metrics that exceed thresholds:
    - Server load > 80%
    - Database performance > 200ms
    - API response time > 150ms
    - Storage capacity > 900GB
    """
    try:
        predictions = predict_all_metrics(hours=hours)
        scaling_alerts = analyze_scaling_needs(predictions)
        
        recommendation = "No immediate scaling needed"
        if scaling_alerts:
            high_severity = [a for a in scaling_alerts if a['severity'] == 'high']
            if high_severity:
                recommendation = "Immediate scaling required for critical metrics"
            else:
                recommendation = "Monitor closely - scaling may be needed soon"
        
        return ScalingAnalysisResponse(
            scaling_alerts=scaling_alerts,
            recommendation=recommendation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crash-risk", response_model=CrashRiskResponse)
async def crash_risk_assessment(hours: int = 24):
    """
    Assess potential crash risk during exam periods
    
    Evaluates risk based on:
    - Server load > 90%
    - Database performance > 400ms
    - API response time > 250ms
    - Storage capacity > 950GB
    """
    try:
        predictions = predict_all_metrics(hours=hours)
        crash_risk = assess_crash_risk(predictions)
        
        return CrashRiskResponse(**crash_risk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/{metric}")
async def predict_metric(metric: str, hours: int = 24):
    """
    Get predictions for a specific metric
    
    Available metrics:
    - server_load: CPU usage percentage
    - db_performance: Database query response time (ms)
    - api_response_time: API response time (ms)
    - storage_capacity: Storage used (GB)
    """
    valid_metrics = ['server_load', 'db_performance', 'api_response_time', 'storage_capacity']
    
    if metric not in valid_metrics:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid metric. Must be one of: {', '.join(valid_metrics)}"
        )
    
    try:
        predictions = predict_all_metrics(hours=hours)
        
        if metric not in predictions:
            raise HTTPException(status_code=404, detail=f"Predictions not available for {metric}")
        
        timestamps = [(datetime.now() + timedelta(hours=i)).isoformat() for i in range(1, hours + 1)]
        
        return {
            "metric": metric,
            "predictions": [
                {"timestamp": ts, "value": val} 
                for ts, val in zip(timestamps, predictions[metric])
            ],
            "summary": {
                "min": float(min(predictions[metric])),
                "max": float(max(predictions[metric])),
                "avg": float(sum(predictions[metric]) / len(predictions[metric]))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/legacy")
async def predict_legacy():
    """Legacy endpoint for backward compatibility"""
    pred = predict_next()
    return {"predicted_cpu_usage": pred}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
