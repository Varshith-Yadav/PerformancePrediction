import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from database.connection import fetch_data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime, timedelta

# Metrics to predict
METRICS = ['server_load', 'db_performance', 'api_response_time', 'storage_capacity']
SEQ_LENGTH = 24

# Load models and scalers
models = {}
scalers = {}

def load_models():
    """Load all trained models and scalers"""
    global models, scalers
    
    models_dir = 'models'
    scalers_dir = os.path.join(models_dir, 'scalers')
    
    for metric in METRICS:
        model_path = os.path.join(models_dir, f'{metric}_model.h5')
        scaler_path = os.path.join(scalers_dir, f'{metric}_scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            models[metric] = load_model(model_path)
            scalers[metric] = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"Model or scaler not found for {metric}. Please train models first.")

def predict_future(model, scaler, last_sequence, hours=24):
    """Predict next N hours for a given metric"""
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(hours):
        pred_input = current_seq.reshape((1, SEQ_LENGTH, 1))
        pred = model.predict(pred_input, verbose=0)
        predictions.append(pred[0, 0])
        current_seq = np.append(current_seq[1:], pred[0, 0])
    
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    return predictions.flatten()

def get_latest_data():
    """Get latest data from database or generate synthetic data"""
    try:
        df = fetch_data()
        # Ensure we have the required columns
        if 'server_load' not in df.columns:
            # If database doesn't have the new columns, generate synthetic data
            return generate_synthetic_data()
        return df
    except:
        # Fallback to synthetic data if database connection fails
        return generate_synthetic_data()

def generate_synthetic_data():
    """Generate synthetic data for testing"""
    np.random.seed(42)
    n_hours = 60 * 24
    time = np.arange(0, n_hours)
    timestamps = pd.date_range(datetime(2024, 1, 1), periods=n_hours, freq='h')
    
    base_load = 40
    daily_pattern = 15 * np.sin(time / 24 * 2 * np.pi)
    weekly_pattern = 10 * np.sin(time / (24 * 7) * 2 * np.pi)
    exam_spikes = np.zeros(n_hours)
    for i in range(0, n_hours, 14 * 24):
        exam_spikes[i:i+72] = 25 + 10 * np.random.random(72)
    
    server_load = np.clip(base_load + daily_pattern + weekly_pattern + exam_spikes + np.random.normal(0, 3, n_hours), 0, 100)
    db_performance = np.clip(50 + (server_load * 0.8) + np.random.normal(0, 5, n_hours), 20, 500)
    api_response_time = np.clip(30 + (server_load * 0.6) + np.random.normal(0, 3, n_hours), 10, 300)
    
    storage_base = 500
    storage_trend = np.linspace(0, 200, n_hours)
    storage_capacity = storage_base + storage_trend
    storage_spikes = np.random.choice(n_hours, size=20, replace=False)
    for spike in storage_spikes:
        storage_capacity[spike:] += np.random.uniform(5, 15)
    storage_capacity = np.clip(storage_capacity, 500, 1000)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'server_load': server_load,
        'db_performance': db_performance,
        'api_response_time': api_response_time,
        'storage_capacity': storage_capacity
    })

def predict_all_metrics(hours=24):
    """Predict all metrics for the next N hours"""
    if not models:
        load_models()
    
    df = get_latest_data()
    predictions = {}
    
    for metric in METRICS:
        if metric not in df.columns:
            continue
            
        # Get last sequence
        values = df[metric].values.reshape(-1, 1)
        scaled = scalers[metric].transform(values)
        last_seq = scaled[-SEQ_LENGTH:]
        
        # Predict
        preds = predict_future(models[metric], scalers[metric], last_seq, hours)
        predictions[metric] = preds.tolist()
    
    return predictions

def analyze_scaling_needs(predictions):
    """Determine when scaling is needed"""
    server_threshold = 80
    db_threshold = 200
    api_threshold = 150
    storage_threshold = 900
    
    scaling_alerts = []
    
    if 'server_load' in predictions:
        pred_array = np.array(predictions['server_load'])
        if pred_array.max() > server_threshold:
            peak_hours = np.where(pred_array > server_threshold)[0].tolist()
            scaling_alerts.append({
                'metric': 'server_load',
                'severity': 'high' if pred_array.max() > 90 else 'medium',
                'peak_hours': peak_hours,
                'max_value': float(pred_array.max()),
                'threshold': server_threshold
            })
    
    if 'db_performance' in predictions:
        pred_array = np.array(predictions['db_performance'])
        if pred_array.max() > db_threshold:
            peak_hours = np.where(pred_array > db_threshold)[0].tolist()
            scaling_alerts.append({
                'metric': 'db_performance',
                'severity': 'high' if pred_array.max() > 300 else 'medium',
                'peak_hours': peak_hours,
                'max_value': float(pred_array.max()),
                'threshold': db_threshold
            })
    
    if 'api_response_time' in predictions:
        pred_array = np.array(predictions['api_response_time'])
        if pred_array.max() > api_threshold:
            peak_hours = np.where(pred_array > api_threshold)[0].tolist()
            scaling_alerts.append({
                'metric': 'api_response_time',
                'severity': 'high' if pred_array.max() > 200 else 'medium',
                'peak_hours': peak_hours,
                'max_value': float(pred_array.max()),
                'threshold': api_threshold
            })
    
    if 'storage_capacity' in predictions:
        pred_array = np.array(predictions['storage_capacity'])
        if pred_array.max() > storage_threshold:
            scaling_alerts.append({
                'metric': 'storage_capacity',
                'severity': 'high',
                'peak_hours': [int(np.argmax(pred_array))],
                'max_value': float(pred_array.max()),
                'threshold': storage_threshold
            })
    
    return scaling_alerts

def assess_crash_risk(predictions):
    """Assess crash risk during exam periods"""
    risk_factors = []
    
    if 'server_load' in predictions and np.array(predictions['server_load']).max() > 90:
        risk_factors.append('critical_server_load')
    
    if 'db_performance' in predictions and np.array(predictions['db_performance']).max() > 400:
        risk_factors.append('critical_db_performance')
    
    if 'api_response_time' in predictions and np.array(predictions['api_response_time']).max() > 250:
        risk_factors.append('critical_api_latency')
    
    if 'storage_capacity' in predictions and np.array(predictions['storage_capacity']).max() > 950:
        risk_factors.append('critical_storage')
    
    risk_score = len(risk_factors) / 4.0
    
    return {
        'risk_score': float(risk_score),
        'risk_level': 'critical' if risk_score > 0.75 else 'high' if risk_score > 0.5 else 'medium' if risk_score > 0.25 else 'low',
        'risk_factors': risk_factors,
        'recommendation': 'Immediate scaling required' if risk_score > 0.75 else 'Monitor closely' if risk_score > 0.5 else 'Normal operations'
    }

def predict_next():
    """Legacy function for backward compatibility"""
    predictions = predict_all_metrics(hours=1)
    if 'server_load' in predictions:
        return float(predictions['server_load'][0])
    return 0.0
