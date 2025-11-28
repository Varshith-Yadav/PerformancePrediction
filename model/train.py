import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from database.connection import fetch_data
import joblib
import os
import tensorflow as tf
from datetime import datetime

# Metrics to predict
METRICS = ['server_load', 'db_performance', 'api_response_time', 'storage_capacity']
SEQ_LENGTH = 24

def generate_synthetic_data():
    """Generate synthetic LMS performance data"""
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

def create_sequences(data, seq_length=SEQ_LENGTH):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_model():
    """Train LSTM models for all metrics"""
    tf.random.set_seed(42)
    
    # Get data
    try:
        df = fetch_data()
        if 'server_load' not in df.columns:
            print("Database doesn't have required columns. Using synthetic data.")
            df = generate_synthetic_data()
    except Exception as e:
        print(f"Database connection failed: {e}. Using synthetic data.")
        df = generate_synthetic_data()
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/scalers', exist_ok=True)
    
    # Create scalers and sequences
    scalers = {}
    sequences = {}
    
    for metric in METRICS:
        if metric not in df.columns:
            continue
            
        scaler = MinMaxScaler()
        values = df[metric].values.reshape(-1, 1)
        scaled = scaler.fit_transform(values)
        scalers[metric] = scaler
        
        X, y = create_sequences(scaled, SEQ_LENGTH)
        sequences[metric] = {'X': X, 'y': y}
    
    # Train models
    for metric in METRICS:
        if metric not in sequences:
            continue
            
        print(f"\nTraining {metric} model...")
        X = sequences[metric]['X']
        y = sequences[metric]['y']
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
            Dropout(0.2),
            LSTM(32, return_sequences=True),
            Dropout(0.2),
            LSTM(16),
            Dropout(0.1),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(f'models/{metric}_best.h5', monitor='val_loss', save_best_only=True)
        ]
        
        # Train
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            verbose=1,
            callbacks=callbacks
        )
        
        # Save model
        model.save(f'models/{metric}_model.h5')
        print(f"✓ {metric} model saved")
    
    # Save scalers
    for metric in METRICS:
        if metric in scalers:
            joblib.dump(scalers[metric], f'models/scalers/{metric}_scaler.pkl')
            print(f"✓ {metric} scaler saved")
    
    return "All models trained and saved successfully!"

